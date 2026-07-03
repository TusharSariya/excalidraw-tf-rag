# deploy/tofu/cnpg.tf
#
# CloudNativePG operator + a single-instance Cluster, backing up to
# SeaweedFS's S3 gateway via the Barman Cloud CNPG-I plugin (the in-tree
# .spec.backup.barmanObjectStore path is deprecated as of CNPG 1.26 and
# removed in 1.31 — verified against
# https://cloudnative-pg.io/plugin-barman-cloud/docs/migration/ on
# 2026-07-02, so we start on the plugin instead of building on deprecated
# ground).
#
# Off-box restore is a P0b acceptance gate (decision log, codex fold #2) —
# this Cluster is what that gate exercises. Single instance (not 3) because
# this is a single-node desktop cluster; PITR/backup is the durability
# story, not replica failover.

resource "helm_release" "cnpg_operator" {
  name             = "cnpg"
  repository       = "https://cloudnative-pg.github.io/charts"
  chart            = "cloudnative-pg"
  version          = var.cnpg_chart_version
  namespace        = var.cnpg_namespace
  create_namespace = true

  timeout = 300
}

# The Barman Cloud CNPG-I plugin has no published helm chart as of
# v0.13.0 (verified via GitHub releases API 2026-07-02) — its documented
# install path is `kubectl apply --server-side -f manifest.yaml` from the
# GitHub release. There is no native OpenTofu/kubernetes-provider resource
# for "apply this multi-document YAML from a URL", so this is a deliberate
# null_resource escape hatch rather than a kubernetes_manifest per object
# (which would require hand-splitting the YAML and tracking plugin-internal
# CRDs/webhooks that change across releases). Re-run `tofu apply` after a
# version bump to reconcile; `tofu destroy` does NOT remove this — see
# deploy/README.md teardown section.
resource "null_resource" "cnpg_barman_plugin" {
  triggers = {
    version         = var.cnpg_barman_plugin_version
    kubeconfig_path = var.kubeconfig_path
    kube_context    = var.kube_context
  }

  # kubeconfig_path defaults to "~/.kube/config" (variables.tf); bash does
  # NOT tilde-expand inside a double-quoted KUBECONFIG="..." assignment, so
  # kubectl silently fell back to localhost:8080 here ("dial tcp
  # 127.0.0.1:8080: connect: connection refused") — hit live on 2026-07-03.
  # pathexpand() resolves "~" to the real home dir before it ever reaches
  # the shell.
  provisioner "local-exec" {
    command = <<-EOT
      set -euo pipefail
      KUBECONFIG="${pathexpand(var.kubeconfig_path)}" kubectl ${var.kube_context != "" ? "--context=${var.kube_context}" : ""} \
        apply --server-side -f \
        "https://github.com/cloudnative-pg/plugin-barman-cloud/releases/download/${var.cnpg_barman_plugin_version}/manifest.yaml"
    EOT
  }

  depends_on = [helm_release.cnpg_operator, helm_release.cert_manager]
}

resource "kubernetes_secret" "cnpg_s3_credentials" {
  metadata {
    name      = "cnpg-seaweedfs-credentials"
    namespace = kubernetes_namespace.platform["storage"].metadata[0].name
  }

  data = {
    ACCESS_KEY_ID     = local.seaweedfs_s3_access_key
    ACCESS_SECRET_KEY = local.seaweedfs_s3_secret_key
  }
}

# ObjectStore CR (barman-cloud.cnpg.io CRD, installed by the plugin manifest
# above). destinationPath / endpointURL point at the in-cluster SeaweedFS S3
# gateway service (seaweedfs.tf).
#
# forcePathStyle REMOVED — confirmed live against this cluster's actual
# ObjectStore CRD (barmancloud.cnpg.io/v1, plugin-barman-cloud v0.13.0) on
# 2026-07-03 via `kubectl explain objectstore.spec.configuration`: there is
# NO such field (checked the plugin's Go types too —
# github.com/cloudnative-pg/barman-cloud pkg/api/config.go has EndpointURL
# but no PathStyle/ForcePathStyle field at all). Setting it made
# `tofu apply` fail with "Manifest configuration incompatible with
# resource schema" / "Unable to find schema type for attribute:
# object.spec.configuration.forcePathStyle". Path-style addressing for a
# custom endpointURL appears to be automatic in this barman-cloud version
# (matches the underlying AWS CLI/boto3 behavior: a non-empty
# --endpoint-url implies path-style) — verify against SeaweedFS S3 access
# logs after backups start flowing if this assumption is wrong.
resource "kubernetes_manifest" "cnpg_backup_store" {
  manifest = {
    apiVersion = "barmancloud.cnpg.io/v1"
    kind       = "ObjectStore"
    metadata = {
      name      = "cnpg-seaweedfs-store"
      namespace = kubernetes_namespace.platform["storage"].metadata[0].name
    }
    spec = {
      configuration = {
        destinationPath = "s3://${var.postgres_backup_bucket}/"
        endpointURL     = "http://seaweedfs-s3.${var.seaweedfs_namespace}.svc.cluster.local:8333"
        s3Credentials = {
          accessKeyId = {
            name = kubernetes_secret.cnpg_s3_credentials.metadata[0].name
            key  = "ACCESS_KEY_ID"
          }
          secretAccessKey = {
            name = kubernetes_secret.cnpg_s3_credentials.metadata[0].name
            key  = "ACCESS_SECRET_KEY"
          }
        }
        wal = {
          compression = "gzip"
        }
        data = {
          compression = "gzip"
        }
      }
    }
  }

  depends_on = [null_resource.cnpg_barman_plugin, helm_release.seaweedfs]
}

resource "kubernetes_manifest" "postgres_cluster" {
  manifest = {
    apiVersion = "postgresql.cnpg.io/v1"
    kind       = "Cluster"
    metadata = {
      name      = "pipeline-v2-pg"
      namespace = kubernetes_namespace.platform["storage"].metadata[0].name
    }
    spec = {
      instances = 1 # single-node desktop cluster; durability via PITR, not replica failover.

      imageName = "ghcr.io/cloudnative-pg/postgresql:16.6"

      storage = {
        size = var.postgres_storage_size
      }

      resources = {
        requests = {
          cpu    = "250m"
          memory = "512Mi"
        }
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }

      plugins = [
        {
          name          = "barman-cloud.cloudnative-pg.io"
          isWALArchiver = true
          parameters = {
            barmanObjectName = kubernetes_manifest.cnpg_backup_store.manifest.metadata.name
          }
        }
      ]
    }
  }

  depends_on = [kubernetes_manifest.cnpg_backup_store]
}
