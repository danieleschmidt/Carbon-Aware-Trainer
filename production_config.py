
# Production service configuration
import os
from typing import Dict, Any

class ProductionConfig:
    """Production configuration for Carbon-Aware-Trainer."""
    
    # Database
    DATABASE_URL = os.getenv(
        'DATABASE_URL', 
        'postgresql://carbon-trainer:password@carbon-trainer-db:5432/carbon_trainer'
    )
    
    # Carbon data sources
    ELECTRICITYMAP_API_KEY = os.getenv('ELECTRICITYMAP_API_KEY')
    WATTTIME_API_KEY = os.getenv('WATTTIME_API_KEY')
    
    # Service configuration
    SERVICE_PORT = int(os.getenv('PORT', 8080))
    SERVICE_HOST = os.getenv('HOST', '0.0.0.0')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Carbon optimization
    CARBON_THRESHOLD = float(os.getenv('CARBON_THRESHOLD', 100.0))
    PAUSE_THRESHOLD = float(os.getenv('PAUSE_THRESHOLD', 150.0))
    RESUME_THRESHOLD = float(os.getenv('RESUME_THRESHOLD', 75.0))
    
    # Scaling
    MIN_REPLICAS = int(os.getenv('MIN_REPLICAS', 2))
    MAX_REPLICAS = int(os.getenv('MAX_REPLICAS', 20))
    AUTO_SCALING = os.getenv('AUTO_SCALING', 'true').lower() == 'true'
    
    # Monitoring
    PROMETHEUS_ENABLED = os.getenv('PROMETHEUS_ENABLED', 'true').lower() == 'true'
    JAEGER_ENABLED = os.getenv('JAEGER_ENABLED', 'true').lower() == 'true'
    
    # Security
    API_KEY_REQUIRED = os.getenv('API_KEY_REQUIRED', 'true').lower() == 'true'
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # Global deployment
    REGIONS = os.getenv('REGIONS', 'us-west-2,us-east-1,eu-west-1').split(',')
    MULTI_REGION_ENABLED = os.getenv('MULTI_REGION_ENABLED', 'true').lower() == 'true'

config = ProductionConfig()
