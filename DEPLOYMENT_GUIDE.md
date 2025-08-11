# Carbon-Aware-Trainer Production Deployment Guide

## 🚀 Production-Ready Carbon-Aware ML Training System

This guide covers deploying the Carbon-Aware-Trainer system in production environments with all three generations of capabilities.

## 📋 System Overview

The Carbon-Aware-Trainer provides three generations of capabilities:

### Generation 1: Advanced Features (Make it Work)
- ✅ Multi-region orchestration and placement optimization
- ✅ Real-time training optimization with carbon awareness
- ✅ Federated learning with carbon-aware client selection
- ✅ Intelligent carbon forecasting and scheduling

### Generation 2: Robustness (Make it Robust)  
- ✅ Circuit breaker pattern for resilient operations
- ✅ Comprehensive validation and security scanning
- ✅ Health monitoring and alerting system
- ✅ Advanced error handling and recovery

### Generation 3: Scaling (Make it Scale)
- ✅ Performance optimization with adaptive resource management
- ✅ Intelligent caching with hybrid eviction policies
- ✅ Predictive auto-scaling with carbon awareness
- ✅ Load balancing and resource pooling

## 🛠️ Installation

### Requirements

- Python 3.8+
- 8GB+ RAM (16GB+ recommended for production)
- Multi-core CPU (4+ cores recommended)
- GPU support (optional, for ML training)

### Basic Installation

```bash
# Install from PyPI (when published)
pip install carbon-aware-trainer

# Or install from source
git clone https://github.com/yourusername/carbon-aware-trainer
cd carbon-aware-trainer
pip install -e ".[all]"
```

### Production Installation

```bash
# Install with all production dependencies
pip install carbon-aware-trainer[all]

# Or with specific components
pip install carbon-aware-trainer[pytorch,monitoring,cloud]
```

## ⚙️ Configuration

### Environment Variables

```bash
# Carbon Data APIs
export ELECTRICITYMAP_API_KEY="your-electricitymap-key"
export WATTTIME_API_KEY="your-watttime-key"

# Default Settings
export CARBON_AWARE_MODE="balanced"        # balanced, carbon_first, performance_first
export CARBON_THRESHOLD="100"              # gCO2/kWh
export CARBON_CHECK_INTERVAL="300"         # seconds

# Multi-region Setup
export CARBON_AWARE_REGIONS="US-CA,US-WA,EU-FR"
export CARBON_MIGRATION_ENABLED="true"

# Performance Settings
export MAX_WORKERS="8"                     # CPU cores to use
export CACHE_SIZE_MB="2048"                # Cache memory limit
export ENABLE_PREDICTIVE_SCALING="true"

# Monitoring
export HEALTH_CHECK_INTERVAL="60"          # seconds
export ALERT_WEBHOOK_URL="https://your-webhook-url"
```

### Configuration File

Create `carbon_config.yaml`:

```yaml
carbon_aware:
  mode: balanced
  threshold: 100
  
  data_sources:
    primary: electricitymap
    fallback: watttime
    cache_duration: 3600
  
  scheduling:
    algorithm: predictive
    flexibility: 0.3
    min_batch_hours: 2
    
  regions:
    preferred:
      - US-WA    # Clean hydro power
      - EU-NO    # Clean hydro power  
      - EU-FR    # Clean nuclear power
    avoid:
      - US-WV    # Coal heavy
      - PL       # Coal heavy
      
  performance:
    max_workers: 8
    cache_size_mb: 2048
    enable_auto_scaling: true
    enable_predictive: true
      
  monitoring:
    health_checks: true
    circuit_breakers: true
    alerting: true
    
  security:
    input_validation: strict
    rate_limiting: true
    audit_logging: true
```

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install carbon-aware-trainer
COPY . .
RUN pip install -e ".[all]"

# Create non-root user
RUN useradd --create-home --shell /bin/bash carbon
USER carbon

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import carbon_aware_trainer; print('OK')"

# Start command
CMD ["python", "-m", "carbon_aware_trainer.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  carbon-trainer:
    build: .
    ports:
      - "8080:8080"
    environment:
      - ELECTRICITYMAP_API_KEY=${ELECTRICITYMAP_API_KEY}
      - WATTTIME_API_KEY=${WATTTIME_API_KEY}
      - CARBON_AWARE_MODE=balanced
      - CARBON_THRESHOLD=100
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
```

## ☸️ Kubernetes Deployment

### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: carbon-aware-trainer
  labels:
    app: carbon-aware-trainer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: carbon-aware-trainer
  template:
    metadata:
      labels:
        app: carbon-aware-trainer
    spec:
      containers:
      - name: carbon-trainer
        image: carbon-aware-trainer:latest
        ports:
        - containerPort: 8080
        env:
        - name: CARBON_AWARE_MODE
          value: "balanced"
        - name: CARBON_THRESHOLD
          value: "100"
        - name: ELECTRICITYMAP_API_KEY
          valueFrom:
            secretKeyRef:
              name: carbon-api-keys
              key: electricitymap-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: carbon-trainer-service
spec:
  selector:
    app: carbon-aware-trainer
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: carbon-trainer-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: carbon-aware-trainer
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'carbon-trainer'
    static_configs:
      - targets: ['carbon-trainer:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard

Import the provided Grafana dashboard (`monitoring/grafana-dashboard.json`) which includes:

- Carbon intensity trends
- Training performance metrics
- System health indicators
- Auto-scaling events
- Circuit breaker status
- Cache hit rates

### Alerting Rules

```yaml
# monitoring/alerts.yml
groups:
- name: carbon-aware-trainer
  rules:
  - alert: HighCarbonIntensity
    expr: carbon_intensity_current > 300
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High carbon intensity detected"
      description: "Carbon intensity is {{ $value }} gCO2/kWh"
      
  - alert: TrainingFailure
    expr: training_failures_total > 5
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Multiple training failures"
      
  - alert: CircuitBreakerOpen
    expr: circuit_breaker_state{state="open"} > 0
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Circuit breaker is open"
```

## 🔒 Security Configuration

### API Keys Management

```bash
# Using environment variables (not recommended for production)
export ELECTRICITYMAP_API_KEY="your-key"

# Using Kubernetes secrets (recommended)
kubectl create secret generic carbon-api-keys \
  --from-literal=electricitymap-key='your-key' \
  --from-literal=watttime-key='your-key'

# Using Docker secrets
echo 'your-key' | docker secret create electricitymap-key -
```

### Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: carbon-trainer-netpol
spec:
  podSelector:
    matchLabels:
      app: carbon-aware-trainer
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS for carbon APIs
```

## 📈 Performance Tuning

### Resource Allocation

```yaml
# For production workloads
resources:
  requests:
    memory: "4Gi"    # Minimum for caching
    cpu: "1000m"     # 1 CPU core minimum
  limits:
    memory: "16Gi"   # Allow for large models
    cpu: "4000m"     # 4 CPU cores maximum
```

### Cache Configuration

```python
# config/cache_settings.py
CACHE_CONFIG = {
    'max_memory_mb': 4096,      # 4GB cache
    'default_ttl': 3600,        # 1 hour TTL
    'enable_compression': True,
    'eviction_policy': 'hybrid_lru_lfu'
}
```

### Auto-scaling Settings

```yaml
# Fine-tuned auto-scaling
autoscaling:
  enabled: true
  min_replicas: 2
  max_replicas: 20
  target_cpu: 70
  target_memory: 80
  scale_up_stabilization: 60s
  scale_down_stabilization: 300s
```

## 🧪 Testing in Production

### Smoke Tests

```bash
# Health check
curl http://your-domain/health

# Carbon data fetch test
curl http://your-domain/api/carbon/current/US-CA

# Training metrics
curl http://your-domain/api/metrics
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://your-domain/api/carbon/current/US-CA

# Using wrk
wrk -t12 -c400 -d30s http://your-domain/api/carbon/forecast/US-CA
```

### Integration Tests

```python
# Run comprehensive integration tests
python -m pytest tests/integration/ -v --production
```

## 🚨 Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check cache usage
   curl http://your-domain/api/cache/stats
   
   # Adjust cache size
   export CACHE_SIZE_MB=1024
   ```

2. **Carbon API Rate Limits**
   ```yaml
   # Increase cache duration
   carbon_aware:
     cache_duration: 7200  # 2 hours
   ```

3. **Circuit Breaker Tripping**
   ```bash
   # Check circuit breaker status
   curl http://your-domain/api/circuit-breakers
   
   # Reset specific breaker
   curl -X POST http://your-domain/api/circuit-breakers/carbon-api/reset
   ```

### Logs Analysis

```bash
# View application logs
kubectl logs -f deployment/carbon-aware-trainer

# Search for errors
kubectl logs deployment/carbon-aware-trainer | grep ERROR

# Monitor in real-time
kubectl logs -f deployment/carbon-aware-trainer --tail=100
```

## 📞 Support & Maintenance

### Health Monitoring

The system includes comprehensive health monitoring:
- `/health` - Basic health check
- `/ready` - Readiness probe
- `/metrics` - Prometheus metrics
- `/api/system/status` - Detailed system status

### Regular Maintenance

1. **Weekly**: Review carbon efficiency metrics
2. **Monthly**: Update carbon intensity thresholds
3. **Quarterly**: Review and update forecasting models
4. **Annually**: Audit carbon reduction achievements

### Backup & Recovery

```bash
# Backup configuration
kubectl get configmap carbon-config -o yaml > backup/config-$(date +%Y%m%d).yaml

# Backup metrics data
pg_dump carbon_metrics > backup/metrics-$(date +%Y%m%d).sql
```

## 🌍 Carbon Impact Reporting

The system automatically generates carbon impact reports:

- Real-time carbon savings dashboard
- Weekly efficiency reports
- Monthly carbon accounting
- Annual sustainability metrics
- ESG compliance reports

## 📚 Additional Resources

- [API Documentation](docs/api.md)
- [Configuration Reference](docs/configuration.md)
- [Performance Tuning Guide](docs/performance.md)
- [Monitoring Setup](docs/monitoring.md)
- [Security Best Practices](docs/security.md)

## 🤝 Support

For production support:
- Documentation: [https://carbon-aware-trainer.org](https://carbon-aware-trainer.org)
- Issues: [GitHub Issues](https://github.com/yourusername/carbon-aware-trainer/issues)
- Community: [Discord Server](https://discord.gg/carbon-aware)
- Enterprise: enterprise@carbon-aware-trainer.org

---

**🌱 Happy Carbon-Aware Training! 🌱**