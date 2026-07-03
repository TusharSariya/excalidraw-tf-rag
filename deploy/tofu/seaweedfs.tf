# deploy/tofu/seaweedfs.tf
#
# SeaweedFS: single-node object store (S3 gateway), no replication
# (master.defaultReplication = "000" is the chart default — verified in
# https://github.com/seaweedfs/seaweedfs/blob/master/k8s/charts/seaweedfs/values.yaml
# on 2026-07-02). One master, one volume server, one filer, one S3 gateway —
# durability is "don't lose the disk", not distributed replication; this is
# a single desktop, not a cluster of them.
#
# S3 credentials are generated here (not chart-default random) so other
# platform components (CNPG barman-cloud backups) can reference them via a
# Kubernetes Secret instead of scraping the chart-generated one.

resource "random_password" "seaweedfs_s3_access_key" {
  length  = 20
  special = false
}

resource "random_password" "seaweedfs_s3_secret_key" {
  length  = 40
  special = false
}

locals {
  seaweedfs_s3_access_key = random_password.seaweedfs_s3_access_key.result
  seaweedfs_s3_secret_key = random_password.seaweedfs_s3_secret_key.result
}

resource "kubernetes_secret" "seaweedfs_s3_config" {
  metadata {
    name      = "seaweedfs-s3-config"
    namespace = kubernetes_namespace.platform["storage"].metadata[0].name
  }

  data = {
    seaweedfs_s3_config = jsonencode({
      identities = [
        {
          name = "admin"
          credentials = [
            {
              accessKey = local.seaweedfs_s3_access_key
              secretKey = local.seaweedfs_s3_secret_key
            }
          ]
          actions = ["Admin", "Read", "Write"]
        }
      ]
    })
  }
}

resource "helm_release" "seaweedfs" {
  name       = "seaweedfs"
  repository = "https://seaweedfs.github.io/seaweedfs/helm"
  chart      = "seaweedfs"
  version    = var.seaweedfs_chart_version
  namespace  = kubernetes_namespace.platform["storage"].metadata[0].name

  # Master: single instance, no replication (chart default "000" retained
  # explicitly for clarity rather than relying on the chart default).
  #
  # master.defaultReplication is pinned type="string": "000" would
  # otherwise be inferred as a number by Helm's set-value type inference
  # and mangled to "0".
  # Structured values (dataDirs is a list of maps) go through yamlencode —
  # `--set`-style list indexing silently mis-renders them; and the previous
  # dataDirs[0].type "hostPathOrPersistentVolumeClaim" was not a real type
  # (it's the *menu* of types), which made the chart render a volumeMount
  # named "data" with no matching volume. PVC on k3s's default local-path
  # StorageClass is the honest single-node choice.
  values = [yamlencode({
    global = { enableSecurity = false }
    master = {
      replicas           = 1
      defaultReplication = "000"
      volumeSizeLimitMB  = 5000
    }
    volume = {
      replicas = 1
      dataDirs = [{
        name         = "data"
        type         = "persistentVolumeClaim"
        size         = var.seaweedfs_volume_size
        storageClass = "local-path"
      }]
    }
    filer = { replicas = 1 }
    s3 = {
      enabled              = true
      replicas             = 1
      existingConfigSecret = kubernetes_secret.seaweedfs_s3_config.metadata[0].name
    }
  })]

  timeout = 300

  depends_on = [kubernetes_secret.seaweedfs_s3_config]
}

# Convenience service alias: the chart creates a service named
# "<release>-s3" in-namespace (verified in values.yaml service-naming
# pattern; NOT confirmed against a live install in this session — check
# `kubectl get svc -n storage` after apply and adjust cnpg.tf's endpointURL
# if the generated name differs).
