#!/bin/bash

# Production Deployment Script for Carbon-Aware Training Platform
# This script automates the deployment process with safety checks

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${1:-production}"
VERSION="${2:-latest}"
DRY_RUN="${3:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in docker kubectl helm curl jq; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Build Docker image
build_image() {
    log_info "Building Docker image for version $VERSION..."
    
    cd "$PROJECT_ROOT"
    
    # Build production image
    if [ "$DRY_RUN" = "false" ]; then
        docker build -f Dockerfile.prod -t "carbon-aware-trainer:$VERSION" .
        docker tag "carbon-aware-trainer:$VERSION" "carbon-aware-trainer:latest"
        log_success "Docker image built successfully"
    else
        log_info "[DRY RUN] Would build docker image carbon-aware-trainer:$VERSION"
    fi
}

# Run tests before deployment
run_tests() {
    log_info "Running pre-deployment tests..."
    
    cd "$PROJECT_ROOT"
    
    if [ "$DRY_RUN" = "false" ]; then
        # Create test environment
        if [ ! -d "test_env" ]; then
            python3 -m venv test_env
        fi
        
        source test_env/bin/activate
        pip install -e . -q
        pip install -r requirements.txt -q
        pip install pytest pytest-asyncio -q
        
        # Run critical tests
        pytest tests/integration/ -v --tb=short
        
        deactivate
        log_success "All tests passed"
    else
        log_info "[DRY RUN] Would run integration tests"
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes cluster..."
    
    local namespace="carbon-training"
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$namespace" &> /dev/null; then
        if [ "$DRY_RUN" = "false" ]; then
            kubectl create namespace "$namespace"
            log_success "Created namespace $namespace"
        else
            log_info "[DRY RUN] Would create namespace $namespace"
        fi
    fi
    
    # Apply configurations
    local k8s_dir="$PROJECT_ROOT/k8s/production"
    
    if [ "$DRY_RUN" = "false" ]; then
        # Apply ConfigMaps and Secrets first
        kubectl apply -f "$k8s_dir/configmap.yml" -n "$namespace" || true
        kubectl apply -f "$k8s_dir/secrets.yml" -n "$namespace" || true
        
        # Apply PVCs
        kubectl apply -f "$k8s_dir/pvc.yml" -n "$namespace" || true
        
        # Apply RBAC
        kubectl apply -f "$k8s_dir/rbac.yml" -n "$namespace" || true
        
        # Apply Deployments and Services
        kubectl apply -f "$k8s_dir/deployment.yml" -n "$namespace"
        kubectl apply -f "$k8s_dir/service.yml" -n "$namespace"
        
        # Apply Ingress
        kubectl apply -f "$k8s_dir/ingress.yml" -n "$namespace" || true
        
        # Apply HPA
        kubectl apply -f "$k8s_dir/hpa.yml" -n "$namespace" || true
        
        log_success "Kubernetes resources deployed"
    else
        log_info "[DRY RUN] Would deploy Kubernetes resources to $namespace"
    fi
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    local namespace="carbon-training"
    local timeout=300  # 5 minutes
    
    if [ "$DRY_RUN" = "false" ]; then
        # Wait for API deployment
        if ! kubectl wait --for=condition=available --timeout="${timeout}s" deployment/carbon-training-api -n "$namespace"; then
            log_error "API deployment failed to become available within $timeout seconds"
            return 1
        fi
        
        # Wait for worker deployment
        if ! kubectl wait --for=condition=available --timeout="${timeout}s" deployment/carbon-training-worker -n "$namespace"; then
            log_error "Worker deployment failed to become available within $timeout seconds"
            return 1
        fi
        
        # Wait for scheduler deployment
        if ! kubectl wait --for=condition=available --timeout="${timeout}s" deployment/carbon-training-scheduler -n "$namespace"; then
            log_error "Scheduler deployment failed to become available within $timeout seconds"
            return 1
        fi
        
        log_success "All deployments are ready"
    else
        log_info "[DRY RUN] Would wait for deployments to be ready"
    fi
}

# Health check
health_check() {
    log_info "Performing health checks..."
    
    local namespace="carbon-training"
    local service_name="carbon-training-service"
    
    if [ "$DRY_RUN" = "false" ]; then
        # Port forward to service for health check
        kubectl port-forward -n "$namespace" "service/$service_name" 8080:80 &
        local port_forward_pid=$!
        
        # Wait for port forward to establish
        sleep 5
        
        # Health check
        local health_url="http://localhost:8080/health"
        local max_retries=10
        local retry=0
        
        while [ $retry -lt $max_retries ]; do
            if curl -f "$health_url" &> /dev/null; then
                log_success "Health check passed"
                kill $port_forward_pid 2>/dev/null || true
                return 0
            fi
            
            retry=$((retry + 1))
            log_warn "Health check failed (attempt $retry/$max_retries), retrying in 10 seconds..."
            sleep 10
        done
        
        kill $port_forward_pid 2>/dev/null || true
        log_error "Health check failed after $max_retries attempts"
        return 1
    else
        log_info "[DRY RUN] Would perform health checks"
    fi
}

# Smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    local namespace="carbon-training"
    
    if [ "$DRY_RUN" = "false" ]; then
        # Get a pod name for testing
        local pod_name
        pod_name=$(kubectl get pods -n "$namespace" -l app=carbon-training,component=api -o jsonpath='{.items[0].metadata.name}')
        
        if [ -n "$pod_name" ]; then
            # Test API endpoints
            kubectl exec -n "$namespace" "$pod_name" -- curl -f http://localhost:8000/health || {
                log_error "Smoke test failed: health endpoint"
                return 1
            }
            
            kubectl exec -n "$namespace" "$pod_name" -- curl -f http://localhost:8000/metrics || {
                log_warn "Metrics endpoint not accessible (may be normal)"
            }
            
            log_success "Smoke tests passed"
        else
            log_error "No API pods found for smoke testing"
            return 1
        fi
    else
        log_info "[DRY RUN] Would run smoke tests"
    fi
}

# Rollback function
rollback() {
    log_warn "Rolling back deployment..."
    
    local namespace="carbon-training"
    
    if [ "$DRY_RUN" = "false" ]; then
        # Rollback deployments
        kubectl rollout undo deployment/carbon-training-api -n "$namespace" || true
        kubectl rollout undo deployment/carbon-training-worker -n "$namespace" || true
        kubectl rollout undo deployment/carbon-training-scheduler -n "$namespace" || true
        
        log_warn "Rollback initiated"
    else
        log_info "[DRY RUN] Would rollback deployments"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary resources..."
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    # Remove test environment
    rm -rf "$PROJECT_ROOT/test_env" 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting Carbon-Aware Training Platform deployment"
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Version: $VERSION"
    log_info "Dry Run: $DRY_RUN"
    
    # Set trap for cleanup
    trap cleanup EXIT
    trap 'rollback; cleanup; exit 1' ERR
    
    # Deployment steps
    check_prerequisites
    build_image
    run_tests
    deploy_kubernetes
    wait_for_deployment
    health_check
    run_smoke_tests
    
    log_success "Deployment completed successfully!"
    log_info "Access the application at: http://your-domain.com"
    log_info "Grafana dashboard: http://your-domain.com/grafana"
    log_info "Prometheus metrics: http://your-domain.com/prometheus"
    
    # Show deployment status
    if [ "$DRY_RUN" = "false" ]; then
        log_info "Deployment status:"
        kubectl get pods -n carbon-training -o wide
        log_info ""
        log_info "Services:"
        kubectl get services -n carbon-training
    fi
}

# Show help
show_help() {
    echo "Carbon-Aware Training Platform Deployment Script"
    echo ""
    echo "Usage: $0 [ENVIRONMENT] [VERSION] [DRY_RUN]"
    echo ""
    echo "Arguments:"
    echo "  ENVIRONMENT  Deployment environment (default: production)"
    echo "  VERSION      Docker image version (default: latest)"
    echo "  DRY_RUN      true/false - perform dry run (default: false)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Deploy latest to production"
    echo "  $0 staging v1.2.0          # Deploy v1.2.0 to staging"
    echo "  $0 production latest true  # Dry run for production"
    echo ""
    echo "Prerequisites:"
    echo "  - docker, kubectl, helm, curl, jq installed"
    echo "  - Docker daemon running"
    echo "  - Kubernetes cluster accessible"
    echo "  - Proper RBAC permissions"
}

# Parse arguments
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    show_help
    exit 0
fi

# Run main function
main "$@"