#!/usr/bin/env python3
"""Production health checks for Carbon-Aware Trainer."""

import asyncio
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, 'src')

from carbon_aware_trainer.core.scheduler import CarbonAwareTrainer
from carbon_aware_trainer.core.types import TrainingConfig


async def run_health_checks():
    """Run comprehensive health checks."""
    print("🏥 PRODUCTION HEALTH CHECKS")
    print("=" * 35)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "overall_status": "healthy"
    }
    
    try:
        # Test 1: Basic initialization
        print("🧪 Testing basic initialization...")
        config = TrainingConfig(
            resource_optimization_enabled=True,
            enable_resilience=True
        )
        
        trainer = CarbonAwareTrainer(
            model=None,
            config=config,
            carbon_model='cached',
            region='US-CA'
        )
        
        await trainer.initialize()
        await trainer.cleanup()
        
        results["checks"]["initialization"] = "passed"
        print("   ✅ Basic initialization: PASSED")
        
        # Test 2: Configuration validation
        print("🧪 Testing configuration validation...")
        if (config.resource_optimization_enabled and 
            config.enable_resilience and
            config.optimization_mode in ['balanced', 'carbon_first', 'performance_first']):
            results["checks"]["configuration"] = "passed"
            print("   ✅ Configuration validation: PASSED")
        else:
            results["checks"]["configuration"] = "failed"
            print("   ❌ Configuration validation: FAILED")
        
        # Test 3: Feature availability
        print("🧪 Testing feature availability...")
        features_available = 0
        total_features = 3
        
        if hasattr(trainer, 'adaptive_training'):
            features_available += 1
            print("   ✅ Generation 1: Available")
        
        if hasattr(trainer, 'resilience_manager'):
            features_available += 1
            print("   ✅ Generation 2: Available")
            
        if hasattr(trainer, 'performance_optimizer'):
            features_available += 1
            print("   ✅ Generation 3: Available")
        
        if features_available == total_features:
            results["checks"]["features"] = "passed"
            print("   ✅ All features available: PASSED")
        else:
            results["checks"]["features"] = "partial"
            print(f"   ⚠️ Features available: {features_available}/{total_features}")
        
    except Exception as e:
        results["checks"]["error"] = str(e)
        results["overall_status"] = "unhealthy"
        print(f"   ❌ Health check failed: {e}")
    
    # Determine overall status
    if all(status == "passed" for status in results["checks"].values() if status != "partial"):
        results["overall_status"] = "healthy"
        print("
✅ OVERALL STATUS: HEALTHY")
    else:
        results["overall_status"] = "degraded"
        print("
⚠️ OVERALL STATUS: DEGRADED")
    
    # Save results
    with open("health_check_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results["overall_status"] == "healthy"


if __name__ == "__main__":
    success = asyncio.run(run_health_checks())
    sys.exit(0 if success else 1)
