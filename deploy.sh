#!/usr/bin/env bash
# Deploy ClinicalMem to Google Cloud Run (both tracks)
# Usage: ./deploy.sh [mcp|a2a|both]
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-central1}"
TARGET="${1:-both}"

deploy_mcp() {
    echo "==> Building and deploying MCP Server..."
    gcloud builds submit \
        --tag "gcr.io/${PROJECT_ID}/clinicalmem-mcp" \
        --gcs-log-dir="gs://${PROJECT_ID}_cloudbuild/logs" \
        -f Dockerfile.mcp .

    gcloud run deploy clinicalmem-mcp \
        --image "gcr.io/${PROJECT_ID}/clinicalmem-mcp" \
        --region "${REGION}" \
        --platform managed \
        --allow-unauthenticated \
        --port 8080 \
        --memory 512Mi \
        --cpu 1 \
        --min-instances 0 \
        --max-instances 3

    echo "==> MCP Server deployed:"
    gcloud run services describe clinicalmem-mcp --region "${REGION}" --format='value(status.url)'
}

deploy_a2a() {
    echo "==> Building and deploying A2A Agent..."
    gcloud builds submit \
        --tag "gcr.io/${PROJECT_ID}/clinicalmem-a2a" \
        --gcs-log-dir="gs://${PROJECT_ID}_cloudbuild/logs" \
        -f Dockerfile.a2a .

    gcloud run deploy clinicalmem-a2a \
        --image "gcr.io/${PROJECT_ID}/clinicalmem-a2a" \
        --region "${REGION}" \
        --platform managed \
        --allow-unauthenticated \
        --port 8080 \
        --memory 512Mi \
        --cpu 1 \
        --min-instances 0 \
        --max-instances 3 \
        --set-env-vars "GOOGLE_API_KEY=${GOOGLE_API_KEY:-}"

    echo "==> A2A Agent deployed:"
    gcloud run services describe clinicalmem-a2a --region "${REGION}" --format='value(status.url)'
}

case "${TARGET}" in
    mcp)  deploy_mcp ;;
    a2a)  deploy_a2a ;;
    both) deploy_mcp; deploy_a2a ;;
    *)    echo "Usage: $0 [mcp|a2a|both]"; exit 1 ;;
esac
