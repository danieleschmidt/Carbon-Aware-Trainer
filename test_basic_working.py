#!/usr/bin/env python3
"""Quick test script to validate basic carbon-aware training functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
from datetime import datetime, timedelta
from carbon_aware_trainer import CarbonAwareTrainer, CarbonMonitor
from carbon_aware_trainer.core.types import CarbonDataSource, TrainingConfig


async def test_basic_functionality():
    """Test basic carbon-aware training functionality."""
    print("🧪 Testing basic Carbon-Aware-Trainer functionality...")
    
    # Test 1: Basic import and initialization
    print("✅ Imports successful")
    
    # Test 2: Basic configuration
    config = TrainingConfig(
        carbon_threshold=100.0,
        pause_threshold=150.0,
        resume_threshold=80.0
    )
    print("✅ Configuration created")
    
    # Test 3: Trainer initialization with cached data source
    trainer = CarbonAwareTrainer(
        carbon_model='cached',
        region='US-CA',
        target_carbon_intensity=100.0,
        config=config,
        api_key='sample_data/sample_carbon_data.json'  # Use sample data
    )
    print("✅ Trainer initialized")
    
    # Test 4: Context manager functionality
    async with trainer:
        print("✅ Trainer context manager works")
        
        # Test 5: Get carbon metrics
        metrics = trainer.get_carbon_metrics()
        print(f"✅ Carbon metrics retrieved: {metrics['session_id']}")
        
        # Test 6: Mock training step
        try:
            result = await trainer.train_step({'data': 'mock_batch'})
            print("✅ Training step executed")
        except NotImplementedError:
            print("⚠️  Training step needs framework integration (expected)")
    
    print("🎉 Basic functionality test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())