# deploy/tofu/variables.tf
#
# Single-node resource budget: this desktop is 32GB RAM / 20-core / RTX 3060
# Ti 8GB, shared with everything else the operator runs (IDE, browser, local
# LLM dev). The PLATFORM layer below (gpu-operator + CNPG + Strimzi/Kafka +
# SeaweedFS + LGTM) is budgeted to ~8GB RAM total, leaving headroom for
# pipeline-v2 workload pods (deploy/k8s/) and everything else on the box.
#
# Budget breakdown (requests, not limits — limits are set higher per
# component to allow bursting; see each *.tf for the split):
#   gpu-operator (device plugin, dcgm-exporter, toolkit, validator): ~512Mi
#   CNPG operator + 1 Cluster instance:                              ~768Mi
#   Strimzi operator + 1 Kafka broker+controller + 1 share coordinator: ~2.5Gi
#   SeaweedFS (master+volume+filer+s3, single replica each):         ~1.5Gi
#   Grafana LGTM (loki+tempo+prometheus+grafana, single-binary each): ~2.25Gi
#   ------------------------------------------------------------------------
#   Total requests:                                                  ~7.5Gi

variable "kubeconfig_path" {
  description = "Path to the kubeconfig written by deploy/bootstrap/install-host.sh."
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "kubeconfig context to use. Empty string = current-context."
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------
variable "namespaces" {
  description = "Namespaces this platform layer owns, and whether they get the GPU-zero ResourceQuota. `inference` is the ONLY namespace where nvidia.com/gpu requests are allowed — gpu-gateway is the sole pod anywhere permitted to request it."
  type = map(object({
    gpu_zero_quota = bool
  }))
  default = {
    pipeline      = { gpu_zero_quota = true }
    inference     = { gpu_zero_quota = false } # gpu-gateway lives here
    storage       = { gpu_zero_quota = true }
    observability = { gpu_zero_quota = true }
    kafka         = { gpu_zero_quota = true }
  }
}

# ---------------------------------------------------------------------------
# GPU Operator
# ---------------------------------------------------------------------------
variable "gpu_operator_chart_version" {
  description = "nvidia/gpu-operator helm chart version. Verified present in https://nvidia.github.io/gpu-operator index.yaml on 2026-07-02."
  type        = string
  default     = "v26.3.3"
}

variable "gpu_operator_namespace" {
  type    = string
  default = "gpu-operator"
}

# ---------------------------------------------------------------------------
# CloudNativePG
# ---------------------------------------------------------------------------
variable "cnpg_chart_version" {
  description = "cloudnative-pg/cloudnative-pg operator helm chart version. Verified in https://cloudnative-pg.github.io/charts index.yaml on 2026-07-02 (appVersion 1.30.0)."
  type        = string
  default     = "0.29.0"
}

variable "cnpg_barman_plugin_version" {
  description = "cloudnative-pg/plugin-barman-cloud release tag (manifest.yaml install). Verified via GitHub releases API on 2026-07-02."
  type        = string
  default     = "v0.13.0"
}

variable "cert_manager_chart_version" {
  description = <<-EOT
    jetstack/cert-manager helm chart version. NOT in the original scaffold
    — discovered live on 2026-07-03: plugin-barman-cloud's manifest.yaml
    (cnpg.tf, null_resource.cnpg_barman_plugin) ships cert-manager.io/v1
    Certificate/Issuer objects (mTLS between CNPG and the CNPG-I plugin
    sidecar) and fails apply with "no matches for kind Certificate/Issuer
    ... ensure CRDs are installed first" without cert-manager already
    running. Verified present in https://charts.jetstack.io index.yaml.
  EOT
  type        = string
  default     = "v1.20.3"
}

variable "cert_manager_namespace" {
  type    = string
  default = "cert-manager"
}

variable "cnpg_namespace" {
  type    = string
  default = "cnpg-system"
}

variable "postgres_storage_size" {
  type    = string
  default = "10Gi"
}

variable "postgres_backup_bucket" {
  description = "SeaweedFS S3 bucket name for CNPG barman-cloud backups."
  type        = string
  default     = "cnpg-backups"
}

# ---------------------------------------------------------------------------
# Kafka / Strimzi
# ---------------------------------------------------------------------------
variable "strimzi_chart_version" {
  description = "strimzi/strimzi-kafka-operator helm chart version, from https://strimzi.io/charts/ index.yaml, verified 2026-07-02 (appVersion 1.1.0, ships Kafka 4.2.x binaries)."
  type        = string
  default     = "1.1.0"
}

variable "kafka_version" {
  description = "Kafka broker version Strimzi will run. Must be one of the versions bundled by the pinned Strimzi operator image (checked at apply time by the operator, not by tofu validate)."
  type        = string
  default     = "4.2.0"
}

variable "kafka_share_group_lock_duration_ms" {
  description = <<-EOT
    D7 (decision log): lock duration >= extract budget (600s+) so a slow PDF
    extract doesn't lose its share-group lock mid-job.
    NOTE: Kafka's group.share.max.record.lock.duration.ms defaults to 60000
    (1 minute) and CLAMPS this value — see kafka-strimzi.tf, we raise the
    max alongside this default. Verified against
    https://kafka.apache.org/42/configuration/broker-configs/ on 2026-07-02.
  EOT
  type        = number
  default     = 600000
}

# ---------------------------------------------------------------------------
# SeaweedFS
# ---------------------------------------------------------------------------
variable "seaweedfs_chart_version" {
  description = "seaweedfs/seaweedfs helm chart version, from https://seaweedfs.github.io/seaweedfs/helm index.yaml, verified 2026-07-02 (appVersion 4.37)."
  type        = string
  default     = "4.37.0"
}

variable "seaweedfs_namespace" {
  type    = string
  default = "storage"
}

variable "seaweedfs_volume_size" {
  type    = string
  default = "50Gi"
}

# ---------------------------------------------------------------------------
# Observability (Grafana LGTM-flavored stack — see observability-lgtm.tf for
# why Mimir is swapped for Prometheus on a single node)
# ---------------------------------------------------------------------------
variable "observability_namespace" {
  type    = string
  default = "observability"
}

variable "loki_chart_version" {
  description = "grafana/loki chart version, https://grafana.github.io/helm-charts index.yaml, verified 2026-07-02 (appVersion 3.6.7)."
  type        = string
  default     = "7.0.0"
}

variable "tempo_chart_version" {
  description = "grafana/tempo chart version, verified 2026-07-02 (appVersion 2.9.0)."
  type        = string
  default     = "1.24.4"
}

variable "grafana_chart_version" {
  description = "grafana/grafana chart version, verified 2026-07-02 (appVersion 12.3.1)."
  type        = string
  default     = "10.5.15"
}

variable "prometheus_chart_version" {
  description = "prometheus-community/prometheus chart version, https://prometheus-community.github.io/helm-charts index.yaml, verified 2026-07-02 (appVersion v3.13.0)."
  type        = string
  default     = "29.14.0"
}

# ---------------------------------------------------------------------------
# Chaos Mesh
# ---------------------------------------------------------------------------
variable "chaos_mesh_chart_version" {
  description = "chaos-mesh/chaos-mesh helm chart version, https://charts.chaos-mesh.org index.yaml, verified 2026-07-03 (`helm search repo chaos-mesh --versions`, latest stable, appVersion 2.8.3)."
  type        = string
  default     = "2.8.3"
}

variable "chaos_mesh_namespace" {
  type    = string
  default = "chaos-mesh"
}
