
# Monitoring Health Check
import requests
import json
from datetime import datetime

def check_monitoring_health():
    checks = {}
    
    # Prometheus
    try:
        response = requests.get('http://prometheus:9090/-/healthy')
        checks['prometheus'] = response.status_code == 200
    except:
        checks['prometheus'] = False
    
    # Grafana
    try:
        response = requests.get('http://grafana:3000/api/health')
        checks['grafana'] = response.status_code == 200
    except:
        checks['grafana'] = False
    
    # AlertManager
    try:
        response = requests.get('http://alertmanager:9093/-/healthy')
        checks['alertmanager'] = response.status_code == 200
    except:
        checks['alertmanager'] = False
    
    return checks

if __name__ == "__main__":
    health = check_monitoring_health()
    print(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "monitoring_health": health,
        "overall_healthy": all(health.values())
    }, indent=2))
