#!/bin/bash
set -e

echo "🚀 Deploying to Kubernetes"
echo "=========================="

# Apply Kubernetes manifests
kubectl apply -f k8s/

# Wait for deployment to be ready
kubectl rollout status deployment/carbon-aware-trainer

# Get service endpoints
kubectl get services

echo "✅ Kubernetes deployment completed!"
