# deploy/tofu/observability-lgtm.tf
#
# Grafana "LGTM" stack (D4: Grafana LGTM chosen over OpenObserve for
# industry-standard tooling) — but NOT via grafana/lgtm-distributed and NOT
# with Mimir, for a documented single-node reason:
#
#   - lgtm-distributed is a distributed-only umbrella chart (no
#     single-binary/monolithic mode) — confirmed via
#     https://github.com/grafana/helm-charts/blob/main/charts/lgtm-distributed/values.yaml
#     and community discussion (grafana/mimir#6211) on 2026-07-02.
#   - Mimir itself has NO monolithic/single-binary mode (unlike Loki and
#     Tempo, which do) — running it "for real" means ~9 separate
#     microservices (distributor, ingester, querier, compactor,
#     store-gateway, ruler, alertmanager, query-frontend, overrides-exporter)
#     which blows the ~8GB total platform RAM budget on metrics alone.
#
# Pragmatic substitution: keep Loki + Tempo (both have real single-binary
# modes and stay small) for the "L" and "T", swap Mimir's "M" for
# prometheus-community/prometheus (single Prometheus server, the thing
# Mimir's ingestion path is itself modeled on) — same PromQL surface,
# same Grafana datasource type, a fraction of the memory. This is a
# deliberate LGTM -> "L T G + Prometheus-for-M" substitution, not a typo;
# revisit if/when this desktop becomes a multi-node cluster and Mimir's
# horizontal scaling actually pays for itself.

resource "helm_release" "loki" {
  name       = "loki"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "loki"
  version    = var.loki_chart_version
  namespace  = kubernetes_namespace.platform["observability"].metadata[0].name

  # The loki chart (v6+) refuses to install without an explicit
  # loki.schemaConfig — there is no default (found live: "You must provide a
  # schema_config for Loki"). tsdb/v13 on filesystem is the canonical
  # single-binary answer. yamlencode because schemaConfig.configs is a list
  # of maps (--set list indexing is fragile). Distributed components and
  # sidecars explicitly zeroed so an upstream default change can't silently
  # reintroduce a microservices footprint.
  values = [yamlencode({
    deploymentMode = "SingleBinary"
    loki = {
      auth_enabled = false
      commonConfig = { replication_factor = 1 }
      storage      = { type = "filesystem" }
      schemaConfig = {
        configs = [{
          from         = "2024-01-01"
          store        = "tsdb"
          object_store = "filesystem"
          schema       = "v13"
          index        = { prefix = "loki_index_", period = "24h" }
        }]
      }
    }
    singleBinary = { replicas = 1 }
    read         = { replicas = 0 }
    write        = { replicas = 0 }
    backend      = { replicas = 0 }
    chunksCache  = { enabled = false }
    resultsCache = { enabled = false }
    lokiCanary   = { enabled = false }
    test         = { enabled = false }
    gateway      = { enabled = false }
  })]

  timeout = 300
}

resource "helm_release" "tempo" {
  name       = "tempo"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "tempo"
  version    = var.tempo_chart_version
  namespace  = kubernetes_namespace.platform["observability"].metadata[0].name

  # grafana/tempo (singular, not tempo-distributed) is single-binary by
  # design — no deploymentMode toggle needed.
  set = [
    {
      name  = "tempo.storage.trace.backend"
      value = "local"
    },
    {
      name  = "persistence.enabled"
      value = "true"
    },
    {
      name  = "persistence.size"
      value = "5Gi"
    },
  ]

  timeout = 300
}

resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "prometheus"
  version    = var.prometheus_chart_version
  namespace  = kubernetes_namespace.platform["observability"].metadata[0].name

  # Single server, no Alertmanager HA, no Thanos sidecar — this is a
  # single-node desktop, not a metrics platform team.
  set = [
    {
      name  = "server.replicaCount"
      value = "1"
    },
    {
      name  = "server.persistentVolume.size"
      value = "8Gi"
    },
    {
      name  = "alertmanager.enabled"
      value = "false"
    },
    {
      name  = "prometheus-pushgateway.enabled"
      value = "false"
    },
    {
      name  = "server.retention"
      value = "15d"
    },
  ]

  timeout = 300
}

resource "helm_release" "grafana" {
  name       = "grafana"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  version    = var.grafana_chart_version
  namespace  = kubernetes_namespace.platform["observability"].metadata[0].name

  set = [
    {
      name  = "persistence.enabled"
      value = "true"
    },
    {
      name  = "persistence.size"
      value = "1Gi"
    },
  ]

  values = [
    yamlencode({
      datasources = {
        "datasources.yaml" = {
          apiVersion = 1
          datasources = [
            {
              name      = "Prometheus"
              type      = "prometheus"
              url       = "http://prometheus-server.${kubernetes_namespace.platform["observability"].metadata[0].name}.svc.cluster.local"
              access    = "proxy"
              isDefault = true
            },
            {
              name   = "Loki"
              type   = "loki"
              url    = "http://loki.${kubernetes_namespace.platform["observability"].metadata[0].name}.svc.cluster.local:3100"
              access = "proxy"
            },
            {
              name   = "Tempo"
              type   = "tempo"
              url    = "http://tempo.${kubernetes_namespace.platform["observability"].metadata[0].name}.svc.cluster.local:3100"
              access = "proxy"
            }
          ]
        }
      }
    })
  ]

  timeout = 300

  depends_on = [helm_release.loki, helm_release.tempo, helm_release.prometheus]
}
