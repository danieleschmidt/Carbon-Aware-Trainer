#!/bin/bash
set -e

echo "🚀 Deploying Carbon-Aware Trainer to Production"
echo "================================================"

# Build Docker image
echo "📦 Building Docker image..."
docker build -t carbon-aware-trainer:production .

# Run production tests
echo "🧪 Running production tests..."
python test_production_readiness.py

# Deploy with Docker Compose
echo "🚀 Starting production services..."
docker-compose up -d

# Wait for services to be ready
echo "⏱️ Waiting for services to be ready..."
sleep 30

# Health checks
echo "🏥 Performing health checks..."
curl -f http://localhost:8080/health || exit 1

echo "✅ Deployment completed successfully!"
echo "📊 Monitoring available at: http://localhost:3000"
echo "🔍 Metrics available at: http://localhost:9090"
