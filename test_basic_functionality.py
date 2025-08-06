#!/usr/bin/env python3
"""
Basic functionality test for Carbon-Aware-Trainer framework.

This test verifies that all core components can be imported and basic
functionality works without external dependencies.
"""

import sys
import asyncio
from pathlib import Path

# Add the source directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    
    try:
        # Core components
        from carbon_aware_trainer import (
            CarbonAwareTrainer, CarbonMonitor, CarbonForecaster,
            TrainingConfig, CarbonIntensity, MetricsCollector
        )
        print("✓ Core imports successful")
        
        # Strategy components
        from carbon_aware_trainer import ThresholdScheduler, AdaptiveScheduler
        print("✓ Strategy imports successful")
        
        # Integration components
        from carbon_aware_trainer import CarbonAwarePyTorchTrainer, CarbonAwareCallback
        print("✓ Integration imports successful")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

async def test_cached_provider():
    """Test cached carbon data provider functionality."""
    print("\nTesting cached carbon data provider...")
    
    try:
        from carbon_aware_trainer.carbon_models.cached import CachedProvider
        
        # Test with sample data
        sample_data_path = Path(__file__).parent / "sample_data" / "sample_carbon_data.json"
        
        provider = CachedProvider(sample_data_path)
        
        async with provider:
            # Test getting current intensity
            intensity = await provider.get_current_intensity("US-CA")
            print(f"✓ Current intensity for US-CA: {intensity.carbon_intensity} gCO2/kWh")
            
            # Test forecast
            from datetime import datetime, timedelta
            forecast = await provider.get_forecast("US-CA", duration=timedelta(hours=3))
            print(f"✓ Forecast data points: {len(forecast.data_points)}")
            
            # Test supported regions
            regions = provider.get_supported_regions()
            print(f"✓ Supported regions: {regions}")
        
        return True
    except Exception as e:
        print(f"✗ Cached provider test failed: {e}")
        return False

async def test_carbon_monitor():
    """Test carbon monitor with cached data."""
    print("\nTesting carbon monitor...")
    
    try:
        from carbon_aware_trainer import CarbonMonitor, TrainingConfig
        from carbon_aware_trainer.core.types import CarbonDataSource
        
        # Use cached data for testing
        sample_data_path = str(Path(__file__).parent / "sample_data" / "sample_carbon_data.json")
        
        monitor = CarbonMonitor(
            regions=["US-CA", "US-WA"],
            data_source=CarbonDataSource.CACHED,
            api_key=sample_data_path,
            update_interval=1
        )
        
        async with monitor:
            # Test getting current intensity
            intensity = await monitor.get_current_intensity("US-CA")
            print(f"✓ Monitor current intensity: {intensity.carbon_intensity} gCO2/kWh")
            
            # Test forecast
            from datetime import timedelta
            forecast = await monitor.get_forecast("US-CA", hours=2)
            print(f"✓ Monitor forecast: {len(forecast.data_points)} data points")
            
            # Test optimal window finding
            window = monitor.find_optimal_window(
                duration_hours=1,
                max_carbon_intensity=100.0,
                preferred_regions=["US-WA"]  # Lower carbon region
            )
            if window:
                print(f"✓ Found optimal window: {window.start_time} - {window.end_time}")
            else:
                print("ℹ No optimal window found (expected with limited test data)")
        
        return True
    except Exception as e:
        print(f"✗ Carbon monitor test failed: {e}")
        return False

async def test_carbon_aware_trainer():
    """Test basic CarbonAwareTrainer functionality."""
    print("\nTesting CarbonAwareTrainer...")
    
    try:
        from carbon_aware_trainer import CarbonAwareTrainer, TrainingConfig
        
        # Create a mock model for testing
        class MockModel:
            def __call__(self, x):
                return {"loss": 0.5}
        
        class MockOptimizer:
            param_groups = [{"lr": 0.001}]
        
        config = TrainingConfig(
            carbon_threshold=100.0,
            pause_threshold=150.0,
            resume_threshold=80.0
        )
        
        sample_data_path = str(Path(__file__).parent / "sample_data" / "sample_carbon_data.json")
        
        trainer = CarbonAwareTrainer(
            model=MockModel(),
            optimizer=MockOptimizer(),
            carbon_model="cached",
            region="US-CA",
            config=config,
            api_key=sample_data_path
        )
        
        async with trainer:
            # Test getting metrics
            metrics = trainer.get_carbon_metrics()
            print(f"✓ Initial metrics: session_id={metrics['session_id']}")
            
            # Test finding optimal window
            window = await trainer.find_optimal_training_window(duration_hours=1)
            if window:
                print(f"✓ Found optimal window: avg intensity {window.avg_carbon_intensity}")
            else:
                print("ℹ No optimal window found")
            
        return True
    except Exception as e:
        print(f"✗ CarbonAwareTrainer test failed: {e}")
        return False

def test_strategy_components():
    """Test strategy components."""
    print("\nTesting strategy components...")
    
    try:
        from carbon_aware_trainer import ThresholdScheduler, AdaptiveScheduler
        from carbon_aware_trainer.core.types import CarbonIntensity
        from datetime import datetime
        
        # Test threshold scheduler
        threshold_scheduler = ThresholdScheduler(carbon_threshold=100.0)
        
        # Create test carbon intensity
        test_intensity = CarbonIntensity(
            region="US-CA",
            timestamp=datetime.now(),
            carbon_intensity=120.0,
            renewable_percentage=60.0
        )
        
        should_pause = threshold_scheduler.should_pause_training(test_intensity)
        print(f"✓ Threshold scheduler pause decision: {should_pause}")
        
        # Test adaptive scheduler
        adaptive_scheduler = AdaptiveScheduler()
        stats = adaptive_scheduler.get_adaptation_stats()
        print(f"✓ Adaptive scheduler initialized: {stats['strategy']}")
        
        return True
    except Exception as e:
        print(f"✗ Strategy test failed: {e}")
        return False

def test_metrics_collector():
    """Test metrics collection functionality."""
    print("\nTesting metrics collector...")
    
    try:
        from carbon_aware_trainer import MetricsCollector
        from datetime import datetime
        
        collector = MetricsCollector("test_session_123")
        
        # Test epoch tracking
        collector.start_epoch(0)
        
        # Log some training steps
        for step in range(5):
            collector.log_training_step(
                step=step,
                loss=0.5 - step * 0.05,
                power_watts=250.0,
                carbon_intensity=100.0 - step * 5,
                duration_ms=1000.0
            )
        
        # End epoch
        epoch_summary = collector.end_epoch(0)
        print(f"✓ Epoch summary: {epoch_summary.steps} steps, avg loss {epoch_summary.avg_loss:.3f}")
        
        # Get session summary
        session_summary = collector.get_session_summary()
        print(f"✓ Session summary: {session_summary.total_steps} total steps")
        
        return True
    except Exception as e:
        print(f"✗ Metrics collector test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("=" * 60)
    print("Carbon-Aware-Trainer Basic Functionality Test")
    print("=" * 60)
    
    all_passed = True
    
    # Import tests
    all_passed &= test_imports()
    
    # Cached provider tests
    all_passed &= await test_cached_provider()
    
    # Carbon monitor tests
    all_passed &= await test_carbon_monitor()
    
    # CarbonAwareTrainer tests
    all_passed &= await test_carbon_aware_trainer()
    
    # Strategy tests
    all_passed &= test_strategy_components()
    
    # Metrics tests
    all_passed &= test_metrics_collector()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Basic functionality is working.")
    else:
        print("❌ Some tests failed. Check the output above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)