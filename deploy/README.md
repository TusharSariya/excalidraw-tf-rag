# deploy/ — platform runbook

Single-node k3s + NVIDIA GPU Operator, Kafka (Strimzi, KRaft, share groups),
SeaweedFS (S3), CloudNativePG, Grafana-flavored LGTM stack. One 32GB/20-core
desktop, RTX 3060 Ti 8GB, solo operator. See
`docs/rearchitecture-v2-decision-log.md` for the decisions behind every
choice here.

Layout:

```
deploy/
  bootstrap/
    install-host.sh        # THE root script — installs k3s + nvidia runtime
    install-user-tools.sh  # no-root: kubectl, helm, opentofu -> ~/.local/bin
  tofu/                    # OpenTofu platform layer (namespaces, operators, Kafka, PG, S3, LGTM)
  k8s/                     # kustomize: gpu-gateway + pipeline-v2 service stubs
  README.md                # this file
```

## 1. Install order

Run these **in order**, each from the repo root unless noted.

### 1a. Host bootstrap (root, interactive — you run this, not an agent)

```bash
sudo deploy/bootstrap/install-host.sh
```

Installs k3s (pinned `v1.35.5+k3s1`, stable channel), wires the NVIDIA
container runtime into k3s's embedded containerd, writes
`~/.kube/config` owned by you, and runs `nvidia-smi` / `kubectl get nodes`
sanity checks. Idempotent — safe to re-run.

### 1b. User tools (no root)

```bash
deploy/bootstrap/install-user-tools.sh
```

Installs `kubectl` (v1.35.6), `helm` (v3.21.2), `tofu` (1.12.2) into
`~/.local/bin`, each checksum-verified against the upstream-published
SHA256 before install. Add `~/.local/bin` to `PATH` if the script warns you
it isn't already there.

### 1c. Platform layer (OpenTofu)

```bash
cd deploy/tofu
tofu init
tofu plan   # review before applying — this creates namespaces, quotas,
            # 5 helm releases, and 4 CRD-backed manifests (Kafka, CNPG,
            # ObjectStore) against a real cluster for the first time
tofu apply
```

Applies, in dependency order tofu resolves automatically:

1. Namespaces (`pipeline`, `inference`, `storage`, `observability`,
   `kafka`, `gpu-operator`, `cnpg-system`) + `requests.nvidia.com/gpu: 0`
   ResourceQuota on every namespace **except `inference`** — gpu-gateway is
   the only pod anywhere allowed to request the GPU.
2. NVIDIA GPU Operator (device plugin, toolkit, dcgm-exporter).
3. Strimzi operator, then a KafkaNodePool + Kafka custom resource (KRaft,
   single combined controller+broker node, share groups enabled).
4. SeaweedFS (single master/volume/filer/S3-gateway, no replication).
5. CloudNativePG operator, the Barman Cloud CNPG-I plugin (installed via a
   `local-exec` `kubectl apply --server-side` — no helm chart exists for it
   yet, see `deploy/tofu/cnpg.tf` header comment), an `ObjectStore` pointed
   at the SeaweedFS S3 gateway, and a single-instance `Cluster`.
6. Grafana-flavored observability: Loki (single-binary), Tempo
   (single-binary), Prometheus (single server — **not** Mimir; see
   `deploy/tofu/observability-lgtm.tf` for why), Grafana with all three
   wired as datasources.

### 1d. In-repo services (kustomize)

```bash
kubectl apply -k deploy/k8s/overlays/dev
```

Applies the gpu-gateway and pipeline-v2 (extract/embed/indexer) Deployment
stubs. **These use placeholder images** (`*:placeholder`) — nothing will
actually run until `tools/pipeline-v2` and a built gpu-gateway image exist.
This step exists so the manifests are exercised by `kubectl apply
--dry-run=client -k` / kustomize build now, not written blind later.

## 2. Verification commands

```bash
# Cluster + GPU
kubectl get nodes -o wide
kubectl get pods -n gpu-operator
kubectl describe node <node> | grep nvidia.com/gpu   # should show Allocatable: 1

# Kafka
kubectl get kafka -n kafka pipeline-v2
kubectl get kafkanodepool -n kafka single-node
kubectl -n kafka exec -it pipeline-v2-single-node-0 -- \
  bin/kafka-configs.sh --bootstrap-server localhost:9092 \
  --entity-type brokers --entity-default --describe | grep share

# Postgres
kubectl get cluster -n storage pipeline-v2-pg
kubectl get backup -n storage
kubectl cnpg status pipeline-v2-pg -n storage   # requires the cnpg kubectl plugin

# SeaweedFS
kubectl get pods -n storage -l app.kubernetes.io/name=seaweedfs
kubectl -n storage port-forward svc/seaweedfs-s3 8333:8333 &
curl http://localhost:8333   # should get an S3-style XML error, not connection refused

# Observability
kubectl get pods -n observability
kubectl -n observability port-forward svc/grafana 3000:80
# open http://localhost:3000, confirm Prometheus/Loki/Tempo datasources

# GPU-zero quota enforcement (should be REJECTED)
kubectl run gpu-test --image=busybox -n pipeline --overrides='{"spec":{"containers":[{"name":"gpu-test","image":"busybox","resources":{"limits":{"nvidia.com/gpu":"1"}}}]}}' --dry-run=server
```

## 3. Acceptance gates (from the decision log — run before trusting this deploy)

- [ ] **Storage torture** (SeaweedFS, decision log codex fold #5): `kill -9`
      the volume server mid-write, verify no silent corruption; fill the
      volume to disk-full and confirm graceful backpressure, not a crash
      loop; run a full restic/backup restore and diff against source.
- [ ] **Off-box restore** (CNPG, codex fold #2 — this is a P0b gate): take a
      CNPG backup, restore it on a machine that is **not** this desktop,
      confirm data integrity end-to-end (not just "cluster came up").
- [ ] **Rollback torture** (codex fold #7): kill the indexer mid-rollback at
      every step; confirm the previous run stays queryable throughout.
- [ ] **Share-group Python-client spike** (codex fold #1, before P2):
      confirm a real Python Kafka client can join `pipeline-v2`'s share
      group, hold a lock for the full 600000ms window under
      `group.share.max.record.lock.duration.ms=900000`, and that a poison
      message routes to DLQ within `group.share.delivery.count.limit`
      attempts (chart default 5, not overridden here — revisit if extract
      needs more retries before DLQ).
- [ ] **GPU exclusivity**: confirm `kubectl run` with a GPU request outside
      `inference` is rejected by the quota (§2 above), AND separately audit
      that nothing on the host runs bare-metal CUDA outside k8s while
      gpu-gateway is up — the quota cannot stop that, only host discipline
      can (decision log codex fold #6).

## 4. Teardown

```bash
cd deploy/tofu
tofu destroy
```

**Does NOT remove:**
- The Barman Cloud CNPG-I plugin's `kubectl apply --server-side` manifest
  (installed via `null_resource.cnpg_barman_plugin`'s `local-exec`
  provisioner — tofu has no destroy-time hook wired for it in this version;
  clean up manually if needed: `kubectl delete -f
  https://github.com/cloudnative-pg/plugin-barman-cloud/releases/download/v0.13.0/manifest.yaml`).
- CRDs installed by the Strimzi and GPU Operator helm charts (helm's
  default behavior — CRDs are not deleted on `helm uninstall` /
  `tofu destroy` to avoid data loss if you reapply).
- PersistentVolumes, if your storage class's reclaim policy is `Retain`.

Full k3s removal (after `tofu destroy`, only if you want the node gone
entirely):

```bash
/usr/local/bin/k3s-uninstall.sh
```

## 5. Config keys verified vs. NOT verified against real docs

Everything below was checked against live upstream documentation on
2026-07-02 (URLs in the relevant `.tf` file's comments). Flagged here so
nothing gets silently trusted:

**Verified:**
- `group.coordinator.rebalance.protocols` valid values
  `[consumer, classic, share, streams]` — Kafka 4.2 broker-configs docs.
- `group.share.record.lock.duration.ms` / `.min.` / `.max.` — same page,
  including the clamping relationship between them.
- `share.record.lock.duration.ms` (no `group.` prefix) is the separate
  dynamic per-group override key — Kafka 4.2 group-configs docs.
- Share groups are "production-ready" as of Kafka 4.2 (release notes).
- `requests.nvidia.com/gpu` is the only valid ResourceQuota key for GPUs
  (no `limits.nvidia.com/gpu` — extended resources disallow overcommit).
- All pinned chart versions (strimzi 1.1.0, cloudnative-pg 0.29.0,
  seaweedfs 4.37.0, gpu-operator v26.3.3, loki 7.0.0, tempo 1.24.4,
  grafana 10.5.15, prometheus 29.14.0) — fetched live from each chart
  repo's `index.yaml`.
- `barmanObjectStore` in-tree CNPG backup is deprecated (1.26+) and removed
  in 1.31 — used the plugin instead, per CNPG's own migration docs.
- lgtm-distributed has no monolithic mode; Mimir specifically has no
  single-binary mode at all — confirmed via grafana/helm-charts source and
  a grafana/mimir maintainer discussion thread.

**NOT verified (flagged honestly, not invented) — check before relying on them:**
- `group.share.max.record.lock.duration.ms=900000` as the value needed to
  un-clamp a 600000ms default: the clamping *relationship* is documented,
  but nobody signed off on 900000ms specifically as the right ceiling —
  it's inferred, not from the decision log. Confirm with the user or a
  live `kafka-configs.sh --describe` before treating it as final.
- `unstable.api.versions.enable` requirement for share groups: some Kafka
  4.0-era docs mention it; nothing checked for 4.2 in this session
  confirmed it's still required or confirmed it's gone. Strimzi's bundled
  Kafka build may differ from what's on kafka.apache.org — check
  `kubectl -n kafka logs` on the broker for `UnsupportedVersionException`
  after apply if share groups don't come up.
- `ObjectStore.spec.configuration.forcePathStyle` field name: the plugin's
  own docs page referenced "forcePathStyle" for non-AWS S3 endpoints but a
  no-cluster session couldn't run `kubectl explain
  objectstore.spec.configuration` to confirm the field exists verbatim on
  the installed CRD. Verify after `tofu apply`, before trusting backups.
- SeaweedFS chart's generated Service name for the S3 gateway (assumed
  `seaweedfs-s3` given release name `seaweedfs` + component `s3`, per the
  chart's naming convention in `values.yaml` — not confirmed against a
  live `kubectl get svc -n storage`). If wrong, fix
  `deploy/tofu/cnpg.tf`'s `endpointURL`.
- Strimzi's Kafka CR `entityOperator.userOperator`/`topicOperator` as empty
  objects (`{}`) enabling them with all defaults — standard Strimzi
  pattern, but not validated against a live Strimzi CRD schema (no cluster
  in this session; `tofu validate` only checks HCL/JSON syntax, not CRD
  conformance).
- Exact share-coordinator internal-topic config key list
  (`share.coordinator.state.topic.replication.factor` /
  `.min.isr`): key *names* verified from the Kafka broker-configs page,
  but the list may be incomplete for a single-broker deployment — Strimzi
  or Kafka may reject startup if another 3-replica-default internal topic
  config was missed. Watch broker startup logs after first apply.

## 6. Acceptance test for this scaffolding itself

```bash
cd deploy/tofu
tofu init -backend=false
tofu validate
```

No cluster required for either command — `tofu init -backend=false` only
downloads providers, `tofu validate` only checks HCL syntax + internal
consistency (references, types), not against a live API server. This was
run clean as part of building this scaffolding (see commit/PR description
for the verbatim output).
