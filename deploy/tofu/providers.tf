# deploy/tofu/providers.tf
#
# Both providers read the same kubeconfig context — the single-node k3s
# cluster produced by deploy/bootstrap/install-host.sh. No remote state
# backend (single operator, single machine): state is a local file,
# deliberately left out of .gitignore exceptions — see deploy/README.md for
# why local state is acceptable here.

provider "kubernetes" {
  config_path    = var.kubeconfig_path
  config_context = var.kube_context
}

provider "helm" {
  kubernetes = {
    config_path    = var.kubeconfig_path
    config_context = var.kube_context
  }
}
