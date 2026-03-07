#!/usr/bin/env bash
# Deploy ClinicalMem to Azure Container Apps (both tracks)
# Usage: ./deploy.sh [mcp|a2a|both]
#
# Prerequisites:
#   az login
#   Docker installed (for local builds)
set -euo pipefail

RG="${AZURE_RESOURCE_GROUP:-clinicalmem-rg}"
LOCATION="${AZURE_LOCATION:-eastus}"
ACR_NAME="${AZURE_ACR_NAME:-clinicalmemacr}"
ENV_NAME="clinicalmem-env"
TARGET="${1:-both}"

ensure_infra() {
    az group create --name "${RG}" --location "${LOCATION}" --output none 2>/dev/null || true

    if ! az acr show --name "${ACR_NAME}" --resource-group "${RG}" &>/dev/null; then
        az acr create --name "${ACR_NAME}" --resource-group "${RG}" \
            --sku Basic --admin-enabled true --output none
    fi

    if ! az containerapp env show --name "${ENV_NAME}" --resource-group "${RG}" &>/dev/null; then
        az containerapp env create --name "${ENV_NAME}" --resource-group "${RG}" \
            --location "${LOCATION}" --output none
    fi

    az acr login --name "${ACR_NAME}"
}

get_acr_creds() {
    ACR_URL=$(az acr show --name "${ACR_NAME}" --resource-group "${RG}" --query loginServer --output tsv)
    ACR_USER=$(az acr credential show --name "${ACR_NAME}" --resource-group "${RG}" --query "username" --output tsv)
    ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --resource-group "${RG}" --query "passwords[0].value" --output tsv)
}

deploy_mcp() {
    local TAG="${ACR_URL}/clinicalmem-mcp:$(git rev-parse --short HEAD 2>/dev/null || echo latest)"
    echo "==> Building MCP Server..."
    docker build -f Dockerfile.mcp -t "${TAG}" .
    docker push "${TAG}"

    if ! az containerapp show --name clinicalmem-mcp --resource-group "${RG}" &>/dev/null; then
        az containerapp create \
            --name clinicalmem-mcp --resource-group "${RG}" \
            --environment "${ENV_NAME}" \
            --image "${TAG}" \
            --registry-server "${ACR_URL}" --registry-username "${ACR_USER}" --registry-password "${ACR_PASSWORD}" \
            --target-port 8080 --ingress external \
            --min-replicas 0 --max-replicas 3 \
            --cpu 0.5 --memory 1Gi \
            --output none
    else
        az containerapp update --name clinicalmem-mcp --resource-group "${RG}" \
            --image "${TAG}" --output none
    fi

    echo "==> MCP Server deployed:"
    az containerapp show --name clinicalmem-mcp --resource-group "${RG}" \
        --query "properties.configuration.ingress.fqdn" --output tsv
}

deploy_a2a() {
    local TAG="${ACR_URL}/clinicalmem-a2a:$(git rev-parse --short HEAD 2>/dev/null || echo latest)"
    echo "==> Building A2A Agent..."
    docker build -f Dockerfile.a2a -t "${TAG}" .
    docker push "${TAG}"

    if ! az containerapp show --name clinicalmem-a2a --resource-group "${RG}" &>/dev/null; then
        az containerapp create \
            --name clinicalmem-a2a --resource-group "${RG}" \
            --environment "${ENV_NAME}" \
            --image "${TAG}" \
            --registry-server "${ACR_URL}" --registry-username "${ACR_USER}" --registry-password "${ACR_PASSWORD}" \
            --target-port 8080 --ingress external \
            --min-replicas 0 --max-replicas 3 \
            --cpu 0.5 --memory 1Gi \
            --env-vars "GOOGLE_API_KEY=${GOOGLE_API_KEY:-}" \
            --output none
    else
        az containerapp update --name clinicalmem-a2a --resource-group "${RG}" \
            --image "${TAG}" --output none
    fi

    echo "==> A2A Agent deployed:"
    az containerapp show --name clinicalmem-a2a --resource-group "${RG}" \
        --query "properties.configuration.ingress.fqdn" --output tsv
}

ensure_infra
get_acr_creds

case "${TARGET}" in
    mcp)  deploy_mcp ;;
    a2a)  deploy_a2a ;;
    both) deploy_mcp; deploy_a2a ;;
    *)    echo "Usage: $0 [mcp|a2a|both]"; exit 1 ;;
esac
