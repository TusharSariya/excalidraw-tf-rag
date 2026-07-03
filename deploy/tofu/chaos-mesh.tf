# deploy/tofu/chaos-mesh.tf
#
# Chaos Mesh: fault-injection platform used by deploy/chaos/ CRs to prove
# the platform layer (gpu-gateway, Kafka, CNPG, SeaweedFS) survives pod
# kills / network partitions before those experiments are relied on.
#
# Chart version 2.8.3 pinned — verified as the latest entry in
# https://charts.chaos-mesh.org index.yaml via
# `helm search repo chaos-mesh --versions` on 2026-07-03 (appVersion 2.8.3
# matches chart version here, chart uses X.Y.Z == appVersion versioning).
#
# Single-node k3s runtime config (CRITICAL — this is the whole reason this
# isn't just `helm install --set-string chaosDaemon.runtime=containerd`):
# k3s embeds its own containerd at a non-default socket path. The chart's
# own default (chaosDaemon.runtime=docker, socketPath=/var/run/docker.sock)
# assumes a Docker-backed cluster and silently fails to attach to any
# container on k3s. Verified live against this cluster on 2026-07-03 via
# `kubectl show values chaos-mesh/chaos-mesh --version 2.8.3` (chaosDaemon
# section documents the containerd override, and gpu-operator.tf already
# established /run/k3s/containerd/containerd.sock as this cluster's actual
# socket path — same k3s install, same path).
#
# dashboard.create=false: no ingress/auth story exists for the dashboard on
# this single-operator desktop cluster yet, and it isn't needed for CR-driven
# experiments — CRs are applied/deleted directly with kubectl. Revisit if
# a human wants a visual chaos console later.
#
# controllerManager.replicaCount=1 / chaosDaemon has no meaningful replica
# knob (it's a DaemonSet, one per node — trivially 1 on this single-node
# cluster): single-node cluster, HA of the controller buys nothing.
#
# clusterScoped stays at the chart default (true): PodChaos/NetworkChaos CRs
# in deploy/chaos/ target pods across inference/kafka/storage namespaces, so
# chaos-mesh's controller needs cluster-wide RBAC to select and act on them.

resource "kubernetes_namespace" "chaos_mesh" {
  metadata {
    name = var.chaos_mesh_namespace
    labels = {
      "app.kubernetes.io/managed-by" = "opentofu"
      "excalidraw-tf-rag/layer"      = "platform"
    }
  }
}

# Same gpu-zero quota pattern as namespaces.tf: chaos-mesh's controller,
# chaos-daemon, and dnsserver pods never request nvidia.com/gpu, so pin that
# explicitly for consistency with every other non-GPU platform namespace
# (belt-and-suspenders — see namespaces.tf header comment).
resource "kubernetes_resource_quota" "chaos_mesh_gpu_zero" {
  metadata {
    name      = "gpu-zero"
    namespace = kubernetes_namespace.chaos_mesh.metadata[0].name
  }

  spec {
    hard = {
      "requests.nvidia.com/gpu" = "0"
    }
  }
}

resource "helm_release" "chaos_mesh" {
  name       = "chaos-mesh"
  repository = "https://charts.chaos-mesh.org"
  chart      = "chaos-mesh"
  version    = var.chaos_mesh_chart_version
  namespace  = kubernetes_namespace.chaos_mesh.metadata[0].name

  # Minimal single-node footprint: 1 controller-manager replica (chart
  # default is 3, meant for multi-node HA clusters), dashboard disabled,
  # chaos-daemon pointed at k3s's real containerd socket instead of the
  # chart's Docker-assuming default.
  set = [
    {
      name  = "controllerManager.replicaCount"
      value = "1"
    },
    {
      name  = "chaosDaemon.runtime"
      value = "containerd"
    },
    {
      name  = "chaosDaemon.socketPath"
      value = "/run/k3s/containerd/containerd.sock"
    },
    {
      name  = "dashboard.create"
      value = "false"
    },
  ]

  timeout = 300

  depends_on = [kubernetes_resource_quota.chaos_mesh_gpu_zero]
}
