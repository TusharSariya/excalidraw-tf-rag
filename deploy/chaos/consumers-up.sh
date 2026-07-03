#!/usr/bin/env bash
# T0g: bring up the 6 pipeline-v2 stage consumers (extract/embed/indexer x
# glr/rlr) as IN-CLUSTER pods. No custom image + no sudo: a stock
# python:3.11-slim pod hostPath-mounts the repo and runs the already-synced
# venv's site-packages via PYTHONPATH (proven to import cleanly). Running in
# the cluster means: real PodChaos targets, direct ClusterIP access to
# Kafka/CNPG/gateway/SeaweedFS, and ENFORCED k8s memory limits (the OOM
# guarantee — extraction can never balloon past its cgroup).
set -euo pipefail
KC="${KUBECTL:-$HOME/.local/bin/kubectl}"
REPO=/home/tushar/excalidraw-tf-rag
NODE_ARTIFACT_ROOT=/home/tushar/excalidraw-tf-rag/pipeline-v2/data/artifacts

# --- connection env (from live secrets) ---
PGURI=$("$KC" get secret pipeline-v2-pg-app -n storage -o jsonpath='{.data.uri}' | base64 -d)
S3AK=$("$KC" get secret cnpg-seaweedfs-credentials -n storage -o jsonpath='{.data.ACCESS_KEY_ID}' | base64 -d)
S3SK=$("$KC" get secret cnpg-seaweedfs-credentials -n storage -o jsonpath='{.data.ACCESS_SECRET_KEY}' | base64 -d)
BOOTSTRAP=pipeline-v2-kafka-bootstrap.kafka.svc:9092
GATEWAY=http://gpu-gateway.inference.svc:8765
S3ENDPOINT=http://seaweedfs-s3.storage.svc:8333

# --- apply work_state schema to the in-cluster DB (idempotent) ---
echo "applying work_state schema to in-cluster CNPG..."
"$KC" run schema-apply --rm -i --restart=Never --image=postgres:16 -n storage \
  --command -- psql "$PGURI" -v ON_ERROR_STOP=1 -f - < "$REPO/pipeline-v2/src/pipeline_v2/schema.sql" 2>&1 | tail -2 || true

# --- credentials secret in pipeline ns ---
"$KC" create secret generic pipeline-v2-conn -n pipeline \
  --from-literal=PIPELINE_V2_DATABASE_URL="$PGURI" \
  --from-literal=PIPELINE_V2_S3_ACCESS_KEY="$S3AK" \
  --from-literal=PIPELINE_V2_S3_SECRET_KEY="$S3SK" \
  --dry-run=client -o yaml | "$KC" apply -f - >/dev/null

# --- per-(stage,corpus) memory limits: extract loads docling (heaviest);
#     embed is a thin HTTP client; indexer writes Lance/tantivy. ---
declare -A MEM=( [extract]=3Gi [embed]=1Gi [indexer]=2Gi )
declare -A REQ=( [extract]=1Gi [embed]=256Mi [indexer]=512Mi )

gen() {  # stage corpus
  local stage=$1 corpus=$2 name="pv2-${stage}-${corpus}"
  local extra_env=""
  if [ "$stage" = "extract" ]; then
    # cap docling concurrency; force CPU (gateway owns the GPU)
    extra_env="- {name: PIPELINE_V2_EXTRACT_WORKERS, value: \"2\"}
        - {name: GRAPH_RAG_DOCLING_DEVICE, value: \"cpu\"}
        - {name: CUDA_VISIBLE_DEVICES, value: \"\"}
        - {name: HF_HUB_OFFLINE, value: \"1\"}
        - {name: TRANSFORMERS_OFFLINE, value: \"1\"}"
  fi
  cat <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${name}
  namespace: pipeline
  labels: {app: pipeline-v2-consumer, stage: "${stage}", corpus: "${corpus}"}
spec:
  replicas: 1
  selector: {matchLabels: {app: pipeline-v2-consumer, stage: "${stage}", corpus: "${corpus}"}}
  template:
    metadata:
      labels: {app: pipeline-v2-consumer, stage: "${stage}", corpus: "${corpus}"}
      annotations: {prometheus.io/scrape: "true", prometheus.io/port: "9464"}
    spec:
      terminationGracePeriodSeconds: 10
      containers:
      - name: consumer
        image: python:3.11-slim
        command: ["python","-m","pipeline_v2.cli","consume","--stage","${stage}","--corpus","${corpus}","--queue","share","--bootstrap-servers","${BOOTSTRAP}"]
        env:
        - {name: PYTHONPATH, value: "/repo/pipeline-v2/src:/repo/pipeline-v2/.venv/lib/python3.11/site-packages"}
        - {name: PYTHONUNBUFFERED, value: "1"}
        - {name: PIPELINE_V2_METRICS_PORT, value: "9464"}
        - {name: PIPELINE_V2_ARTIFACT_BACKEND, value: "s3"}
        - {name: PIPELINE_V2_S3_ENDPOINT, value: "${S3ENDPOINT}"}
        - {name: PIPELINE_V2_S3_BUCKET, value: "pipeline-v2-${corpus}"}
        - {name: PIPELINE_GATEWAY_URL, value: "${GATEWAY}"}
        - {name: PIPELINE_V2_ARTIFACT_ROOT, value: "/repo/pipeline-v2/data/artifacts"}
        ${extra_env}
        envFrom: [{secretRef: {name: pipeline-v2-conn}}]
        volumeMounts: [{name: repo, mountPath: /repo}]
        ports: [{containerPort: 9464, name: metrics}]
        resources:
          requests: {memory: "${REQ[$stage]}", cpu: "100m"}
          limits: {memory: "${MEM[$stage]}"}
      volumes:
      - name: repo
        hostPath: {path: ${REPO}, type: Directory}
EOF
}

echo "generating + applying 6 consumers..."
for stage in extract embed indexer; do
  for corpus in glr rlr; do
    gen "$stage" "$corpus" | "$KC" apply -f -
  done
done

echo "waiting for consumers to be Ready..."
"$KC" wait --for=condition=available deploy -l app=pipeline-v2-consumer -n pipeline --timeout=180s || true
"$KC" get pods -n pipeline -l app=pipeline-v2-consumer -o wide
