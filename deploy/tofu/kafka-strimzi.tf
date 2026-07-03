# deploy/tofu/kafka-strimzi.tf
#
# Apache Kafka 4.x, KRaft mode, single node (combined broker+controller
# role), via the Strimzi operator. Share groups (KIP-932) enabled per
# decision D7 (lock >= extract budget, idempotent artifacts absorb
# duplicates) and the codex fold requiring a Python-client spike before P2
# (see deploy/README.md — this Kafka instance is what that spike runs
# against).
#
# Every config key below was checked against live Apache Kafka
# documentation on 2026-07-02:
#   - group.coordinator.rebalance.protocols: broker config, type list,
#     default "classic,consumer,streams", valid values
#     [consumer, classic, share, streams].
#     Source: https://kafka.apache.org/42/configuration/broker-configs/
#   - group.share.record.lock.duration.ms: broker-level DEFAULT lock
#     duration for share groups, default 30000ms.
#     Source: same page.
#   - group.share.min.record.lock.duration.ms / max.record.lock.duration.ms:
#     bound the above AND any per-share-group override. Defaults 15000 /
#     60000ms. Source: same page.
#   - share.record.lock.duration.ms (no "group." prefix): the DYNAMIC,
#     per-group override config (kafka-configs.sh --entity-type groups),
#     separate from the broker-level "group.share.*" defaults.
#     Source: https://kafka.apache.org/42/configuration/group-configs/
#
# IMPORTANT — flagged, not invented: decision D7 wants a 600000ms (10 min)
# lock duration, but group.share.max.record.lock.duration.ms defaults to
# 60000ms (1 minute) and CLAMPS both the broker default and any per-group
# override above it. We raise the max here to 900000ms (15 min) so the
# requested 600000ms default is actually honored — this was NOT explicit
# in the decision log and needs a sign-off, it's inferred from the
# clamping relationship documented on the broker-configs page above.
#
# Kafka 4.2 release notes (https://kafka.apache.org/blog/2026/02/17/
# apache-kafka-4.2.0-release-announcement/) state share groups are
# "production-ready" as of 4.2 — the `unstable.api.versions.enable=true`
# flag some 4.0-era docs mention for share groups was NOT required in
# anything checked for 4.2 in this session; if Strimzi's bundled Kafka
# build turns out to differ, that flag is the first thing to check.
#
# Share-partition/coordinator topic replication factors default to 3 in
# upstream Kafka; a single broker cannot satisfy that, so they're pinned to
# 1 below (single-node-cluster adjustment — verified as the correct override
# key names, NOT verified as a complete list against a live single-broker
# Strimzi deploy in this session).
#
# apiVersion kafka.strimzi.io/v1 (NOT v1beta2): this Strimzi build
# (strimzi-kafka-operator chart 1.1.0) only registers v1 for the
# Kafka/KafkaNodePool CRDs — confirmed live via
# `kubectl get crd kafkas.kafka.strimzi.io -o jsonpath='{.spec.versions[*].name}'`
# on this cluster on 2026-07-03, which returned only "v1". v1beta2 was the
# name in older Strimzi docs/examples; using it here 404s at the API
# server ("cannot select exact GV from REST mapper").

resource "helm_release" "strimzi_operator" {
  name             = "strimzi-kafka-operator"
  repository       = "https://strimzi.io/charts/"
  chart            = "strimzi-kafka-operator"
  version          = var.strimzi_chart_version
  namespace        = kubernetes_namespace.platform["kafka"].metadata[0].name
  create_namespace = false

  set = [
    {
      name  = "watchAnyNamespace"
      value = "false"
    },
  ]

  timeout = 300
}

resource "kubernetes_manifest" "kafka_node_pool" {
  manifest = {
    apiVersion = "kafka.strimzi.io/v1"
    kind       = "KafkaNodePool"
    metadata = {
      name      = "single-node"
      namespace = kubernetes_namespace.platform["kafka"].metadata[0].name
      labels = {
        "strimzi.io/cluster" = "pipeline-v2"
      }
    }
    spec = {
      replicas = 1
      # Combined controller+broker role: only sane choice for a one-node
      # cluster (KRaft still needs >=1 controller; splitting controller
      # and broker into separate pools on one physical node buys nothing).
      roles = ["controller", "broker"]
      storage = {
        type = "persistent-claim"
        size = "20Gi"
      }
      resources = {
        requests = {
          cpu    = "500m"
          memory = "1536Mi"
        }
        limits = {
          cpu    = "2"
          memory = "2560Mi"
        }
      }
    }
  }

  depends_on = [helm_release.strimzi_operator]
}

resource "kubernetes_manifest" "kafka_cluster" {
  manifest = {
    apiVersion = "kafka.strimzi.io/v1"
    kind       = "Kafka"
    metadata = {
      name      = "pipeline-v2"
      namespace = kubernetes_namespace.platform["kafka"].metadata[0].name
      annotations = {
        "strimzi.io/node-pools" = "enabled"
        "strimzi.io/kraft"      = "enabled"
      }
    }
    spec = {
      kafka = {
        version = var.kafka_version
        listeners = [
          {
            name = "plain"
            port = 9092
            type = "internal"
            tls  = false
          }
        ]
        config = {
          "offsets.topic.replication.factor"         = 1
          "transaction.state.log.replication.factor" = 1
          "transaction.state.log.min.isr"            = 1
          "default.replication.factor"               = 1
          "min.insync.replicas"                      = 1

          # --- Share groups (KIP-932) ---
          "group.coordinator.rebalance.protocols" = "classic,consumer,share"

          "group.share.record.lock.duration.ms"     = var.kafka_share_group_lock_duration_ms
          "group.share.min.record.lock.duration.ms" = 15000
          # See header comment: raised above the 60000ms upstream default so
          # a 600000ms per-group override (D7) isn't clamped down.
          "group.share.max.record.lock.duration.ms" = 900000

          # Single-broker cluster: internal share-coordinator state topic
          # replication defaults (3) exceed our broker count.
          "share.coordinator.state.topic.replication.factor" = 1
          "share.coordinator.state.topic.min.isr"            = 1
        }
      }
      entityOperator = {
        topicOperator = {}
        userOperator  = {}
      }
    }
  }

  depends_on = [kubernetes_manifest.kafka_node_pool]
}
