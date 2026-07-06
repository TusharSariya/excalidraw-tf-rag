# deploy/tofu/namespaces.tf
#
# One namespace per platform concern, each with a hard requests.nvidia.com/gpu
# quota of 0 EXCEPT `inference` — gpu-gateway (deploy/k8s/base/gpu-gateway)
# is the only pod anywhere in the cluster allowed to request a GPU. This is
# belt-and-suspenders: the ResourceQuota stops any pod scheduled via the
# k8s API from requesting the resource; bare-metal CUDA loads outside k8s
# are a host-discipline problem this quota cannot solve (see decision log,
# codex fold #6).

resource "kubernetes_namespace" "platform" {
  for_each = var.namespaces

  metadata {
    name = each.key
    labels = {
      "app.kubernetes.io/managed-by" = "opentofu"
      "excalidraw-tf-rag/layer"      = "platform"
    }
  }
}

resource "kubernetes_resource_quota" "gpu_zero" {
  for_each = { for name, cfg in var.namespaces : name => cfg if cfg.gpu_zero_quota }

  metadata {
    name      = "gpu-zero"
    namespace = kubernetes_namespace.platform[each.key].metadata[0].name
  }

  spec {
    hard = {
      # Only "requests.<extended-resource>" is a valid ResourceQuota key for
      # extended resources like GPUs — Kubernetes disallows overcommit for
      # them, so there is no separate "limits.nvidia.com/gpu" quota key.
      # Verified: https://kubernetes.io/docs/concepts/policy/resource-quotas/
      # ("Resource Quota For Extended Resources").
      "requests.nvidia.com/gpu" = "0"
    }
  }
}

# gpu-operator's own namespace is created by its helm chart (create_namespace
# = true in gpu-operator.tf) and deliberately has NO gpu-zero quota: its
# device-plugin/toolkit/dcgm-exporter DaemonSets are the ones that mediate
# GPU access for everyone else and must be able to see the device.
