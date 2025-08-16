#!/usr/bin/env python3
"""
Minimal test without external dependencies to verify basic structure.
"""

import sys
import os
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic imports without async functionality."""
    print("Testing basic imports...")
    
    try:
        # Test core type imports
        from carbon_aware_trainer.core.types import (
            CarbonIntensity, TrainingConfig, TrainingMetrics
        )
        print("✓ Core types imported successfully")
        
        # Test basic scheduler
        from carbon_aware_trainer.strategies.threshold import ThresholdScheduler
        print("✓ Threshold scheduler imported successfully")
        
        # Test configuration 
        config = TrainingConfig(
            carbon_threshold=100.0,
            pause_threshold=150.0,
            resume_threshold=75.0
        )
        print(f"✓ TrainingConfig created: threshold={config.carbon_threshold}")
        
        # Test threshold scheduler
        from datetime import datetime
        scheduler = ThresholdScheduler(carbon_threshold=100.0)
        
        intensity = CarbonIntensity(
            region="US-CA",
            timestamp=datetime.now(),
            carbon_intensity=120.0,
            renewable_percentage=60.0
        )
        
        should_pause = scheduler.should_pause_training(intensity)
        print(f"✓ Scheduler decision: should_pause={should_pause}")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_sample_data():
    """Test sample data loading."""
    print("\nTesting sample data...")
    
    try:
        import json
        sample_path = Path(__file__).parent / "sample_data" / "sample_carbon_data.json"
        
        if sample_path.exists():
            with open(sample_path) as f:
                data = json.load(f)
            print(f"✓ Sample data loaded: {len(data)} regions")
            
            # Validate data structure
            for region, region_data in data.items():
                if 'current' in region_data and 'forecast' in region_data:
                    print(f"✓ {region}: valid structure")
                else:
                    print(f"✗ {region}: invalid structure")
                    return False
            
            return True
        else:
            print(f"✗ Sample data not found at {sample_path}")
            return False
            
    except Exception as e:
        print(f"✗ Sample data test failed: {e}")
        return False

def test_metrics_basic():
    """Test metrics collection without async."""
    print("\nTesting basic metrics...")
    
    try:
        from carbon_aware_trainer.monitoring.metrics import MetricsCollector
        
        collector = MetricsCollector("test_session")
        collector.start_epoch(0)
        
        # Log training steps
        for step in range(3):
            collector.log_training_step(
                step=step,
                loss=1.0 - step * 0.1,
                power_watts=250.0,
                carbon_intensity=100.0,
                duration_ms=1000.0
            )
        
        epoch_summary = collector.end_epoch(0)
        print(f"✓ Epoch completed: {epoch_summary.steps} steps")
        
        session_summary = collector.get_session_summary()
        print(f"✓ Session summary: {session_summary.total_steps} steps")
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False

def main():
    """Run minimal tests."""
    print("=" * 50)
    print("Carbon-Aware-Trainer Minimal Test")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_basic_imports()
    all_passed &= test_sample_data()
    all_passed &= test_metrics_basic()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL MINIMAL TESTS PASSED!")
        print("✓ Core functionality is working")
    else:
        print("❌ Some tests failed")
    print("=" * 50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())