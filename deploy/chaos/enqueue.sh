#!/usr/bin/env bash
# Run `pipeline-v2 enqueue` as a one-shot IN-CLUSTER pod (host can't reach the
# ClusterIP Kafka; a pod on the cluster network can). PDF paths land as
# /repo/... which the extract consumers (same hostPath mount) can read.
# Usage: enqueue.sh <corpus> <limit>   (glr|rlr)
set -euo pipefail
KC="${KUBECTL:-$HOME/.local/bin/kubectl}"
CORPUS="$1"; LIMIT="${2:-5}"; OFFSET="${3:-0}"
case "$CORPUS" in
  glr) GLOB='/repo/graph-layout-rag/data/raw/pdf/*.pdf' ;;
  rlr) GLOB='/repo/rag-literature-rag/data/raw/pdf/*.pdf' ;;
  *) echo "unknown corpus $CORPUS"; exit 1 ;;
esac
NAME="enq-${CORPUS}-$(date +%s)"
BOOTSTRAP=pipeline-v2-kafka-bootstrap.kafka.svc:9092
"$KC" apply -f - >/dev/null 2>&1 <<EOF
apiVersion: v1
kind: Pod
metadata: {name: ${NAME}, namespace: pipeline}
spec:
  restartPolicy: Never
  containers:
  - name: enq
    image: python:3.11-slim
    command: ["python","-m","pipeline_v2.cli","enqueue","--corpus","${CORPUS}","--limit","${LIMIT}","--start-offset","${OFFSET}","--pdf-glob","${GLOB}","--bootstrap-servers","${BOOTSTRAP}"]
    env:
    - {name: PYTHONPATH, value: "/repo/pipeline-v2/src:/repo/pipeline-v2/.venv/lib/python3.11/site-packages"}
    - {name: PYTHONUNBUFFERED, value: "1"}
    volumeMounts: [{name: repo, mountPath: /repo}]
    resources: {limits: {memory: "1Gi"}}
  volumes:
  - name: repo
    hostPath: {path: /home/tushar/excalidraw-tf-rag, type: Directory}
EOF
# wait for completion (bounded)
for _ in $(seq 1 40); do
  ph=$("$KC" get pod "$NAME" -n pipeline -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
  [ "$ph" = "Succeeded" -o "$ph" = "Failed" ] && break
  sleep 3
done
echo "enqueue $CORPUS limit=$LIMIT -> phase=$ph"
"$KC" logs "$NAME" -n pipeline 2>&1 | tail -5
"$KC" delete pod "$NAME" -n pipeline --wait=false >/dev/null 2>&1
