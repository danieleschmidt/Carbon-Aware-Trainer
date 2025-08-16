#!/bin/bash
# Comprehensive health validation for Carbon-Aware-Trainer

set -e

echo "🏥 Carbon-Aware-Trainer Health Validation"
echo "============================================"

# Check service endpoints
echo "Checking service endpoints..."
curl -f http://localhost:8080/health || exit 1
curl -f http://localhost:8080/ready || exit 1
echo "✅ Service endpoints healthy"

# Check database connectivity
echo "Checking database connectivity..."
pg_isready -h localhost -p 5432 || exit 1
echo "✅ Database connectivity OK"

# Check carbon data sources
echo "Checking carbon data sources..."
curl -f "https://api.electricitymap.org/health" || echo "⚠️ ElectricityMap API warning"
curl -f "https://api2.watttime.org/health" || echo "⚠️ WattTime API warning"
echo "✅ Carbon data sources checked"

# Check resource usage
echo "Checking resource usage..."
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

echo "Memory usage: ${MEMORY_USAGE}%"
echo "CPU usage: ${CPU_USAGE}%"

if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
    echo "❌ High memory usage: ${MEMORY_USAGE}%"
    exit 1
fi

if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "❌ High CPU usage: ${CPU_USAGE}%"
    exit 1
fi

echo "✅ Resource usage within limits"

# Check log for errors
echo "Checking application logs..."
if grep -i "error\|exception\|failed" /var/log/carbon-trainer.log | tail -10; then
    echo "⚠️ Recent errors found in logs"
else
    echo "✅ No recent errors in logs"
fi

# Test carbon optimization
echo "Testing carbon optimization..."
python3 -c "
import requests
import json

# Test carbon intensity endpoint
response = requests.get('http://localhost:8080/carbon/US-CA')
if response.status_code == 200:
    data = response.json()
    print(f'✅ Carbon intensity API: {data["carbon_intensity"]} gCO2/kWh')
else:
    print('❌ Carbon intensity API failed')
    exit(1)

# Test training optimization
optimization_request = {
    'duration_hours': 4,
    'max_carbon_intensity': 100,
    'preferred_regions': ['US-WA', 'EU-NO']
}

response = requests.post(
    'http://localhost:8080/training/optimize',
    json=optimization_request,
    headers={'Authorization': 'Bearer test-token'}
)

if response.status_code == 200:
    data = response.json()
    print(f'✅ Training optimization: {data["recommended_region"]}')
else:
    print('❌ Training optimization failed')
    exit(1)
"

echo "✅ Carbon optimization working"

# Performance benchmark
echo "Running performance benchmark..."
python3 -c "
import time
import concurrent.futures
import requests

def make_request():
    start = time.time()
    response = requests.get('http://localhost:8080/health')
    duration = time.time() - start
    return response.status_code == 200, duration

# Test concurrent requests
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(make_request) for _ in range(50)]
    results = [future.result() for future in futures]

success_count = sum(1 for success, _ in results if success)
avg_duration = sum(duration for _, duration in results) / len(results)

print(f'Performance benchmark: {success_count}/50 requests successful')
print(f'Average response time: {avg_duration*1000:.1f}ms')

if success_count < 45:
    print('❌ Poor success rate')
    exit(1)

if avg_duration > 0.5:
    print('❌ High response time')
    exit(1)

print('✅ Performance benchmark passed')
"

echo ""
echo "🎉 All health checks passed!"
echo "Carbon-Aware-Trainer is production ready"
