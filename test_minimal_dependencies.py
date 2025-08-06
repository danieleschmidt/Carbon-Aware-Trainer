#!/usr/bin/env python3
"""
Test script to verify that carbon-aware-trainer works with minimal dependencies.

This script tests the core functionality without optional dependencies like:
- psutil (system metrics)
- PyYAML (YAML config files)
- aiohttp (HTTP requests)
"""

import sys
import asyncio
import json
import tempfile
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that core modules can be imported without optional dependencies."""
    print("Testing basic imports...")
    
    try:
        from carbon_aware_trainer.core.config import ConfigManager, ProductionConfig
        print("✓ ConfigManager imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ConfigManager: {e}")
        return False
    
    try:
        from carbon_aware_trainer.core.metrics_collector import MetricsCollector
        print("✓ MetricsCollector imported successfully") 
    except Exception as e:
        print(f"✗ Failed to import MetricsCollector: {e}")
        return False
        
    try:
        from carbon_aware_trainer.core.api_manager import APIManager
        print("✓ APIManager imported successfully")
    except Exception as e:
        print(f"✗ Failed to import APIManager: {e}")
        return False
        
    try:
        from carbon_aware_trainer.core.alerting import AlertingManager
        print("✓ AlertingManager imported successfully")
    except Exception as e:
        print(f"✗ Failed to import AlertingManager: {e}")
        return False
    
    return True

def test_config_manager():
    """Test ConfigManager with JSON configuration (no YAML dependency)."""
    print("\nTesting ConfigManager with JSON...")
    
    try:
        from carbon_aware_trainer.core.config import ConfigManager, ProductionConfig
        
        # Create a temporary directory for config
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            
            # Create ConfigManager
            config_manager = ConfigManager(config_dir)
            
            # Test loading default config
            config = config_manager.load_config()
            print(f"✓ Default config loaded: environment={config.environment}")
            
            # Test creating JSON config file
            json_config = {
                "environment": "test",
                "carbon_threshold": 120.0,
                "data_source": "electricitymap",
                "log_level": "INFO"
            }
            
            json_file = config_dir / "test.json"
            with open(json_file, 'w') as f:
                json.dump(json_config, f, indent=2)
            
            # Test loading JSON config
            loaded_config = config_manager.load_config(json_file)
            print(f"✓ JSON config loaded: threshold={loaded_config.carbon_threshold}")
            
            return True
            
    except Exception as e:
        print(f"✗ ConfigManager test failed: {e}")
        return False

async def test_metrics_collector():
    """Test MetricsCollector without psutil."""
    print("\nTesting MetricsCollector without psutil...")
    
    try:
        from carbon_aware_trainer.core.metrics_collector import MetricsCollector, HAS_PSUTIL
        
        print(f"  psutil available: {HAS_PSUTIL}")
        
        # Create metrics collector
        collector = MetricsCollector(collection_interval=1, retention_hours=1)
        
        # Test recording custom metrics
        collector.record_metric("test.metric", 42.0, {"category": "test"})
        print("✓ Custom metric recorded")
        
        # Start collector briefly
        await collector.start()
        
        # Let it run for a short time
        await asyncio.sleep(0.1)
        
        # Test metrics summary
        summary = collector.get_metrics_summary(1)
        print(f"✓ Metrics summary generated: {summary['data_points_collected']} points")
        
        # Stop collector
        await collector.stop()
        print("✓ MetricsCollector test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ MetricsCollector test failed: {e}")
        return False

async def test_api_manager():
    """Test APIManager without aiohttp."""
    print("\nTesting APIManager without aiohttp...")
    
    try:
        from carbon_aware_trainer.core.api_manager import APIManager, APIEndpoint, HAS_AIOHTTP
        from carbon_aware_trainer.core.exceptions import CarbonProviderError
        
        print(f"  aiohttp available: {HAS_AIOHTTP}")
        
        # Create API manager
        api_manager = APIManager()
        
        # Test start/stop
        await api_manager.start()
        print("✓ APIManager started")
        
        if not HAS_AIOHTTP:
            # Test that API calls fail gracefully
            result = await api_manager.call_api(
                provider="test",
                endpoint=APIEndpoint.CURRENT_INTENSITY,
                url="http://example.com"
            )
            
            if not result.success and "aiohttp not installed" in str(result.error):
                print("✓ API call failed gracefully without aiohttp")
            else:
                print(f"✗ API call didn't fail as expected: {result.error}")
                return False
        
        # Test statistics
        stats = api_manager.get_call_statistics()
        print(f"✓ Statistics retrieved: {stats['total_calls']} total calls")
        
        await api_manager.stop()
        print("✓ APIManager test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ APIManager test failed: {e}")
        return False

async def test_alerting_manager():
    """Test AlertingManager without aiohttp."""
    print("\nTesting AlertingManager...")
    
    try:
        from carbon_aware_trainer.core.alerting import AlertingManager, AlertType, AlertSeverity, HAS_AIOHTTP
        
        print(f"  aiohttp available: {HAS_AIOHTTP}")
        
        # Create alerting manager
        alerting_manager = AlertingManager()
        
        # Test creating an alert
        alert = await alerting_manager.create_alert(
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            source="test_script"
        )
        
        print(f"✓ Alert created: {alert.alert_id}")
        
        # Test getting active alerts
        active_alerts = alerting_manager.get_active_alerts()
        print(f"✓ Active alerts retrieved: {len(active_alerts)} alerts")
        
        # Test statistics
        stats = alerting_manager.get_alert_statistics()
        print(f"✓ Alert statistics: {stats['active_alerts']} active")
        
        print("✓ AlertingManager test completed")
        
        return True
        
    except Exception as e:
        print(f"✗ AlertingManager test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Carbon Aware Trainer with Minimal Dependencies")
    print("=" * 60)
    
    all_passed = True
    
    # Test basic imports
    if not test_basic_imports():
        all_passed = False
    
    # Test config manager
    if not test_config_manager():
        all_passed = False
    
    # Test metrics collector
    if not await test_metrics_collector():
        all_passed = False
    
    # Test API manager
    if not await test_api_manager():
        all_passed = False
    
    # Test alerting manager
    if not await test_alerting_manager():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Core functionality works with minimal dependencies.")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)