# deploy/tofu/gpu-operator.tf
#
# NVIDIA GPU Operator: manages the device plugin, DCGM exporter, container
# toolkit, and node feature discovery that let pods request nvidia.com/gpu.
# Runs its own namespace (no GPU-zero quota — see namespaces.tf).
#
# k3s uses an embedded containerd at a non-standard path, so the toolkit
# subchart needs to be pointed at k3s's config file + socket explicitly.
# Verified against:
#   https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html
#   ("K3s" install section, CONTAINERD_CONFIG / CONTAINERD_SOCKET env vars)
# on 2026-07-02.
#
# driver.enabled=false: the host's NVIDIA driver is installed out-of-band
# by deploy/bootstrap/install-host.sh (desktop already runs CUDA workloads
# outside k8s — gpu-gateway, local LLM dev — so a host driver must exist
# regardless of the operator).

resource "kubernetes_namespace" "gpu_operator" {
  metadata {
    name = var.gpu_operator_namespace
    labels = {
      "app.kubernetes.io/managed-by" = "opentofu"
      "excalidraw-tf-rag/layer"      = "platform"
    }
  }
}

resource "helm_release" "gpu_operator" {
  name       = "gpu-operator"
  repository = "https://nvidia.github.io/gpu-operator"
  chart      = "gpu-operator"
  version    = var.gpu_operator_chart_version
  namespace  = kubernetes_namespace.gpu_operator.metadata[0].name

  # Single GPU, single node: nothing here needs to be highly available and
  # everything should stay well inside the 8GB RAM platform budget's slack.
  #
  # toolkit.env[3].value is deliberately pinned type="string": the
  # ClusterPolicy CRD's spec.toolkit.env[].value field is string-typed, but
  # Helm's `--set`-style type inference turns a bare "true" into a boolean,
  # which the CRD's OpenAPI schema then rejects at admission
  # ("spec.toolkit.env[3].value: Invalid value: \"boolean\": ... must be of
  # type string") — hit live against this cluster on 2026-07-03.
  set = [
    {
      name  = "driver.enabled"
      value = "false"
    },
    {
      name  = "toolkit.env[0].name"
      value = "CONTAINERD_CONFIG"
    },
    {
      name  = "toolkit.env[0].value"
      value = "/var/lib/rancher/k3s/agent/etc/containerd/config.toml"
    },
    {
      name  = "toolkit.env[1].name"
      value = "CONTAINERD_SOCKET"
    },
    {
      name  = "toolkit.env[1].value"
      value = "/run/k3s/containerd/containerd.sock"
    },
    {
      name  = "toolkit.env[2].name"
      value = "CONTAINERD_RUNTIME_CLASS"
    },
    {
      name  = "toolkit.env[2].value"
      value = "nvidia"
    },
    {
      name  = "toolkit.env[3].name"
      value = "CONTAINERD_SET_AS_DEFAULT"
    },
    {
      name  = "toolkit.env[3].value"
      value = "true"
      type  = "string"
    },
  ]

  timeout         = 600
  wait            = false # DaemonSets need the real GPU node; nothing to wait for at `tofu apply` on first run before the node is labeled.
  atomic          = false
  cleanup_on_fail = true
}
