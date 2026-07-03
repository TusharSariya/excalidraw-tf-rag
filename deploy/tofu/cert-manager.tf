# deploy/tofu/cert-manager.tf
#
# cert-manager: NOT in the original platform design — added live on
# 2026-07-03 after cnpg.tf's null_resource.cnpg_barman_plugin failed to
# apply the plugin-barman-cloud manifest.yaml with "no matches for kind
# Certificate/Issuer in version cert-manager.io/v1 ... ensure CRDs are
# installed first". The CNPG-I barman-cloud plugin uses cert-manager to
# mint the mTLS certs for its gRPC sidecar connection to CNPG (a
# selfsigned Issuer + two Certificates, "barman-cloud-client" and
# "barman-cloud-server", both in cnpg-system — see the plugin's
# manifest.yaml for the exact objects). installCRDs=true so this single
# helm_release is sufficient (no separate CRD-YAML apply step, unlike
# Strimzi/CNPG which ship CRDs via their own charts already).
#
# wait=true (chart default) deliberately: cnpg.tf's barman-plugin
# null_resource depends on this release, and needs the cert-manager
# webhook actually serving before the plugin manifest (which relies on
# cert-manager's admission webhook to validate its Certificate/Issuer
# objects) can apply cleanly.

resource "kubernetes_namespace" "cert_manager" {
  metadata {
    name = var.cert_manager_namespace
    labels = {
      "app.kubernetes.io/managed-by" = "opentofu"
      "excalidraw-tf-rag/layer"      = "platform"
    }
  }
}

resource "helm_release" "cert_manager" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  version    = var.cert_manager_chart_version
  namespace  = kubernetes_namespace.cert_manager.metadata[0].name

  set = [
    {
      name  = "crds.enabled"
      value = "true"
    },
  ]

  timeout = 300
}
