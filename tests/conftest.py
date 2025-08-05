"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path

from carbon_aware_trainer.core.types import CarbonIntensity, CarbonIntensityUnit


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_carbon_intensity():
    """Sample carbon intensity for testing."""
    return CarbonIntensity(
        region="US-CA",
        timestamp=datetime.now(),
        carbon_intensity=120.0,
        unit=CarbonIntensityUnit.GRAMS_CO2_PER_KWH,
        data_source="test",
        renewable_percentage=45.0,
        confidence=0.85
    )


@pytest.fixture
def sample_carbon_data():
    """Sample carbon data for multiple regions and times."""
    regions = ["US-CA", "US-WA", "EU-FR", "EU-DE"]
    base_time = datetime.now()
    
    data = {"regions": {}}
    
    for region in regions:
        historical = []
        forecast = []
        
        # Different base intensities for each region
        base_intensities = {
            "US-CA": 100,
            "US-WA": 60,   # Hydro power - lower carbon
            "EU-FR": 80,   # Nuclear + renewables
            "EU-DE": 140   # More coal - higher carbon
        }
        
        base_intensity = base_intensities[region]
        
        # Generate 24 hours historical + 24 hours forecast
        for i in range(48):
            timestamp = base_time - timedelta(hours=24-i) if i < 24 else base_time + timedelta(hours=i-24)
            
            # Add daily pattern and some randomness
            hour = timestamp.hour
            daily_variation = 20 * abs(0.5 - (hour / 24))  # Higher during day
            random_variation = (i % 7) * 5  # Some pseudo-random variation
            
            intensity = base_intensity + daily_variation + random_variation
            renewable_pct = max(20, min(80, 60 - (intensity - base_intensity) / 2))
            
            data_point = {
                "timestamp": timestamp.isoformat(),
                "carbon_intensity": round(intensity, 1),
                "renewable_percentage": round(renewable_pct, 1),
                "confidence": 0.8 + (i % 3) * 0.05
            }
            
            if i < 24:
                historical.append(data_point)
            else:
                forecast.append(data_point)
        
        data["regions"][region] = {
            "historical": historical,
            "forecast": forecast
        }
    
    return data


@pytest.fixture
def temp_carbon_data_file(sample_carbon_data):
    """Create temporary file with sample carbon data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_carbon_data, f, indent=2)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def mock_training_model():
    """Mock ML model for testing."""
    class MockModel:
        def __init__(self):
            self.step_count = 0
            self.loss_values = [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3]
        
        def train_step(self, batch):
            """Simulate training step with decreasing loss."""
            loss = self.loss_values[min(self.step_count, len(self.loss_values) - 1)]
            accuracy = min(0.95, 0.5 + (self.step_count * 0.05))
            
            self.step_count += 1
            
            return {
                "loss": loss,
                "accuracy": accuracy,
                "learning_rate": 0.001
            }
        
        def __call__(self, batch):
            """Forward pass for simple model usage."""
            return {"output": f"processed_{batch.get('input', 'data')}"}
    
    return MockModel()


@pytest.fixture
def clean_carbon_data():
    """Carbon data with consistently low intensity for testing optimal conditions."""
    base_time = datetime.now()
    
    data = {"regions": {"US-CLEAN": {"historical": [], "forecast": []}}}
    
    for i in range(48):
        timestamp = base_time - timedelta(hours=24-i) if i < 24 else base_time + timedelta(hours=i-24)
        
        # Low, stable carbon intensity
        intensity = 40 + (i % 3) * 5  # 40-50 range
        renewable_pct = 85 + (i % 2) * 5  # 85-90% renewable
        
        data_point = {
            "timestamp": timestamp.isoformat(),
            "carbon_intensity": intensity,
            "renewable_percentage": renewable_pct,
            "confidence": 0.95
        }
        
        if i < 24:
            data["regions"]["US-CLEAN"]["historical"].append(data_point)
        else:
            data["regions"]["US-CLEAN"]["forecast"].append(data_point)
    
    return data


@pytest.fixture
def high_carbon_data():
    """Carbon data with consistently high intensity for testing pause conditions."""
    base_time = datetime.now()
    
    data = {"regions": {"US-DIRTY": {"historical": [], "forecast": []}}}
    
    for i in range(48):
        timestamp = base_time - timedelta(hours=24-i) if i < 24 else base_time + timedelta(hours=i-24)
        
        # High carbon intensity
        intensity = 200 + (i % 5) * 20  # 200-280 range
        renewable_pct = 15 + (i % 3) * 5  # 15-25% renewable
        
        data_point = {
            "timestamp": timestamp.isoformat(),
            "carbon_intensity": intensity,
            "renewable_percentage": renewable_pct,
            "confidence": 0.9
        }
        
        if i < 24:
            data["regions"]["US-DIRTY"]["historical"].append(data_point)
        else:
            data["regions"]["US-DIRTY"]["forecast"].append(data_point)
    
    return data