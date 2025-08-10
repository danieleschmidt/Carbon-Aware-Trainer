#!/usr/bin/env python3
"""Simple PyTorch training example with carbon-awareness."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from carbon_aware_trainer import CarbonAwareTrainer
from carbon_aware_trainer.core.types import TrainingConfig


class MockModel:
    """Mock PyTorch model for demonstration."""
    def __init__(self):
        self.step_count = 0
    
    def train_step(self, batch, **kwargs):
        """Simulate a training step."""
        self.step_count += 1
        loss = 1.0 - (self.step_count * 0.01)  # Simulate decreasing loss
        return {'loss': max(0.1, loss), 'accuracy': min(0.99, self.step_count * 0.01)}


async def main():
    """Run carbon-aware training example."""
    print("🌱 Starting Carbon-Aware PyTorch Training Example")
    
    # Create mock model and optimizer
    model = MockModel()
    optimizer = None  # Mock optimizer
    
    # Configure carbon-aware training
    config = TrainingConfig(
        carbon_threshold=100.0,  # gCO2/kWh
        pause_threshold=150.0,   # Pause above this threshold
        resume_threshold=75.0,   # Resume below this threshold
        check_interval=60        # Check every minute
    )
    
    # Initialize carbon-aware trainer
    trainer = CarbonAwareTrainer(
        model=model,
        optimizer=optimizer,
        carbon_model='cached',  # Use cached sample data
        region='US-CA',
        config=config,
        api_key='sample_data/sample_carbon_data.json'
    )
    
    # Add callback for state changes
    def on_state_change(state, metrics):
        print(f"🔄 Training state: {state.value}, Carbon: {metrics.total_carbon_kg:.3f} kg CO2")
    
    trainer.add_state_callback(on_state_change)
    
    # Run training session
    async with trainer:
        await trainer.start_training()
        
        print("🚀 Starting training loop...")
        
        # Simulate training for 10 steps
        for step in range(10):
            batch = {'data': f'batch_{step}', 'size': 32}
            
            try:
                result = await trainer.train_step(batch)
                
                if step % 5 == 0:
                    metrics = trainer.get_carbon_metrics()
                    print(f"Step {step}: Loss {result['loss']:.3f}, "
                          f"Accuracy {result['accuracy']:.3f}, "
                          f"Carbon {metrics['total_carbon_kg']:.3f} kg CO2")
                
                # Small delay between steps
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error in step {step}: {e}")
                break
        
        # Final metrics
        final_metrics = trainer.get_carbon_metrics()
        print("\n📊 Final Training Metrics:")
        print(f"  Total Steps: {trainer.step}")
        print(f"  Runtime: {final_metrics['runtime_hours']:.2f} hours")
        print(f"  Total Energy: {final_metrics['total_energy_kwh']:.4f} kWh")
        print(f"  Total Carbon: {final_metrics['total_carbon_kg']:.3f} kg CO2")
        print(f"  Average Carbon Intensity: {final_metrics['avg_carbon_intensity']:.1f} gCO2/kWh")
    
    print("✅ Carbon-aware training completed!")


if __name__ == "__main__":
    asyncio.run(main())