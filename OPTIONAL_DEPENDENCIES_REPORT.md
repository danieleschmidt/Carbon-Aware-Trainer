# Optional Dependencies Implementation Report

## Overview

This report documents the implementation of graceful handling for optional dependencies in the Carbon Aware Trainer production robustness modules. The core functionality now works with minimal dependencies while providing helpful error messages when optional features are unavailable.

## Dependencies Made Optional

### 1. psutil (System Metrics)
- **Module**: `src/carbon_aware_trainer/core/metrics_collector.py`
- **Purpose**: Collects system performance metrics (CPU, memory, disk, network)
- **Fallback**: Returns zero-value metrics when psutil is not available
- **Impact**: System monitoring still works but without real system metrics

```python
# Graceful import handling
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

# Fallback implementation
if not HAS_PSUTIL:
    logger.debug("psutil not available, skipping system metrics collection")
    # Create minimal fallback metrics with zero values
```

### 2. PyYAML (YAML Configuration Files)
- **Modules**: 
  - `src/carbon_aware_trainer/core/config.py`
  - `src/carbon_aware_trainer/core/security.py`
- **Purpose**: Supports YAML configuration files
- **Fallback**: Uses JSON configuration files only
- **Impact**: Configuration still works but only supports JSON format

```python
# Graceful import handling
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

# Fallback configuration file priority
if HAS_YAML:
    candidates = ["env.yaml", "env.yml", "env.json", ...]
else:
    candidates = ["env.json", "env.yaml", ...]  # JSON first, YAML fails gracefully
```

### 3. aiohttp (HTTP Client)
- **Modules**:
  - `src/carbon_aware_trainer/core/api_manager.py`
  - `src/carbon_aware_trainer/core/alerting.py`
- **Purpose**: Makes HTTP requests for carbon data APIs and notifications
- **Fallback**: APIs fail gracefully with helpful error messages
- **Impact**: Carbon data fetching and webhook/Slack notifications disabled

```python
# Graceful import handling
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False

# Fallback API call implementation
if not self._http_available:
    result.error = CarbonProviderError(
        "HTTP functionality not available (aiohttp not installed)", 
        provider
    )
    return result
```

## Additional Dependencies Handled

### 4. Pydantic (Data Validation)
- **Module**: `src/carbon_aware_trainer/core/validation.py`
- **Purpose**: Data validation and type checking
- **Fallback**: Simple fallback implementations for validators
- **Impact**: Validation is skipped when pydantic is not available

### 5. GPUtil (GPU Metrics)
- **Module**: `src/carbon_aware_trainer/core/metrics_collector.py`
- **Purpose**: GPU utilization metrics
- **Fallback**: GPU metrics set to None (already implemented with try/catch)
- **Impact**: No GPU metrics collected

## Core Functionality Preserved

### ✅ What Works Without Optional Dependencies

1. **Basic Configuration Management**
   - JSON configuration files supported
   - Environment variable override
   - Configuration validation (basic)

2. **Metrics Collection**
   - Custom metrics recording
   - Metrics aggregation and export
   - Basic performance tracking

3. **API Management**
   - Rate limiting logic
   - Circuit breaker patterns
   - Response caching
   - Statistics tracking

4. **Alerting System**
   - Alert creation and management
   - Email notifications (SMTP only)
   - Alert rules and cooldown periods

5. **Production Features**
   - Backup and recovery logic
   - Security validation
   - Lifecycle management
   - Health checks (basic)

### ❌ What Requires Optional Dependencies

1. **System Metrics**: Requires `psutil`
   - CPU, memory, disk, network usage
   - Real-time system monitoring

2. **YAML Configuration**: Requires `PyYAML`
   - YAML config file support
   - Only JSON configs work without it

3. **HTTP Operations**: Requires `aiohttp`
   - Carbon intensity data fetching
   - Webhook notifications
   - Slack notifications
   - External API integrations

4. **Advanced Validation**: Requires `pydantic`
   - Complex data type validation
   - Schema enforcement

## Installation Options

### Minimal Installation
```bash
pip install carbon-aware-trainer
# Only core dependencies: requests, numpy, pandas, pydantic, python-dateutil, pytz, asyncio-mqtt
```

### Full Installation
```bash
pip install carbon-aware-trainer[all]
# Includes all optional dependencies for full functionality
```

### Selective Installation
```bash
# For system monitoring
pip install carbon-aware-trainer psutil

# For YAML config support
pip install carbon-aware-trainer PyYAML

# For HTTP functionality
pip install carbon-aware-trainer aiohttp

# For GPU metrics
pip install carbon-aware-trainer GPUtil
```

## Testing

A comprehensive test suite verifies that core functionality works without optional dependencies:

```bash
python3 test_minimal_dependencies.py
```

### Test Results
- ✅ Basic imports work without optional dependencies
- ✅ ConfigManager works with JSON configs
- ✅ MetricsCollector works without psutil (using fallback metrics)
- ✅ APIManager fails gracefully without aiohttp
- ✅ AlertingManager works for basic alert management

## Error Messages

When optional functionality is accessed without required dependencies, users see helpful error messages:

- **No psutil**: "psutil not available, skipping system metrics collection"
- **No PyYAML**: "YAML support not available (install PyYAML), cannot load .yaml file"
- **No aiohttp**: "HTTP functionality not available (aiohttp not installed)"
- **No aiohttp (webhooks)**: "Cannot send webhook notifications: aiohttp not available (install aiohttp)"
- **No aiohttp (Slack)**: "Cannot send Slack notifications: aiohttp not available (install aiohttp)"

## Benefits

1. **Lower Barrier to Entry**: Users can install and use core functionality without heavy dependencies
2. **Flexible Deployment**: Choose only the features you need
3. **Resource Efficiency**: Smaller memory footprint in minimal installations
4. **Development Friendly**: Easier testing and development with minimal dependencies
5. **Production Ready**: Graceful degradation ensures stability even if optional features fail

## Recommendations

1. **For Development**: Use minimal installation for core logic development
2. **For Testing**: Full installation to test all features
3. **For Production**: Install only needed optional dependencies based on use case
4. **For CI/CD**: Consider separate test stages for minimal vs full installations

This implementation ensures that Carbon Aware Trainer remains robust and accessible while providing optional advanced features for users who need them.