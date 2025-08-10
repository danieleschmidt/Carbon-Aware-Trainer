#!/usr/bin/env python3
"""Advanced carbon-aware training example with realistic power modeling."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, timedelta
import time
from carbon_aware_trainer import CarbonAwareTrainer, CarbonMonitor
from carbon_aware_trainer.core.types import TrainingConfig


class RealisticMLModel:
    """More realistic ML model with power consumption simulation."""
    
    def __init__(self, model_size_mb=100, gpu_power_watts=250):
        self.model_size_mb = model_size_mb
        self.gpu_power_watts = gpu_power_watts
        self.step_count = 0
        self.total_flops = 0
    
    def train_step(self, batch, **kwargs):
        """Simulate realistic training step with power consumption."""
        self.step_count += 1
        
        # Simulate computational work (affects power draw)
        batch_size = batch.get('size', 32)
        sequence_length = batch.get('seq_len', 512)
        
        # Estimate FLOPs for transformer-like model
        flops_per_token = self.model_size_mb * 1000  # Rough estimate
        step_flops = batch_size * sequence_length * flops_per_token
        self.total_flops += step_flops
        
        # Simulate training time (more complex batches take longer)
        complexity_factor = (batch_size * sequence_length) / 16384  # Normalize
        base_time = 0.1  # Base 100ms per step
        step_time = base_time * max(0.5, complexity_factor)
        time.sleep(step_time)
        
        # Calculate dynamic power based on utilization
        utilization = min(1.0, complexity_factor)
        actual_power_watts = self.gpu_power_watts * (0.3 + 0.7 * utilization)
        
        # Store power info for carbon calculations
        step_info = {
            'loss': max(0.01, 2.0 - (self.step_count * 0.1)),
            'accuracy': min(0.98, self.step_count * 0.05),
            'batch_size': batch_size,
            'step_time_seconds': step_time,
            'gpu_power_watts': actual_power_watts,
            'flops': step_flops,
            'utilization': utilization
        }
        
        return step_info


async def main():
    """Run advanced carbon-aware training with realistic power modeling."""
    print("🌍 Advanced Carbon-Aware Training Example")
    print("==========================================")
    
    # Create realistic model (simulate a 350M parameter transformer)
    model = RealisticMLModel(model_size_mb=350, gpu_power_watts=300)
    
    # Advanced training configuration
    config = TrainingConfig(
        carbon_threshold=120.0,  # Stricter threshold
        pause_threshold=180.0,   # Pause during peak hours
        resume_threshold=90.0,   # Resume during clean periods
        check_interval=30,       # Check every 30 seconds
        preferred_regions=['US-WA', 'US-OR', 'EU-NO']  # Prefer hydro regions
    )
    
    # Initialize enhanced carbon-aware trainer
    trainer = CarbonAwareTrainer(
        model=model,
        carbon_model='cached',
        region='US-CA',
        config=config,
        api_key='sample_data/sample_carbon_data.json'
    )
    
    # Enhanced callbacks
    def detailed_state_callback(state, metrics):
        print(f"\n🔄 State Change: {state.value}")
        print(f"   💡 Energy: {metrics.total_energy_kwh:.4f} kWh")
        print(f"   🌱 Carbon: {metrics.total_carbon_kg:.3f} kg CO2")
        print(f"   ⏱️  Paused: {metrics.paused_duration.total_seconds():.0f}s")
    
    trainer.add_state_callback(detailed_state_callback)
    
    # Training scenarios with varying complexity
    training_scenarios = [
        {'batch_size': 16, 'seq_len': 256, 'description': 'Light load'},
        {'batch_size': 32, 'seq_len': 512, 'description': 'Medium load'}, 
        {'batch_size': 64, 'seq_len': 512, 'description': 'Heavy load'},
        {'batch_size': 32, 'seq_len': 1024, 'description': 'Long sequences'},
        {'batch_size': 8, 'seq_len': 2048, 'description': 'Very long sequences'}
    ]
    
    print(f"🎯 Training Configuration:")
    print(f"   Model: {model.model_size_mb}MB ({model.gpu_power_watts}W GPU)")
    print(f"   Carbon Threshold: {config.carbon_threshold} gCO2/kWh")
    print(f"   Scenarios: {len(training_scenarios)} different workloads")
    
    async with trainer:
        await trainer.start_training()
        
        print("\n🚀 Starting Advanced Training Session...")
        
        total_scenarios = len(training_scenarios) * 4  # 4 steps per scenario
        completed_steps = 0
        
        for scenario_idx, scenario in enumerate(training_scenarios):
            print(f"\n📋 Scenario {scenario_idx + 1}: {scenario['description']}")
            print(f"   Batch: {scenario['batch_size']}, Seq Len: {scenario['seq_len']}")
            
            # Run 4 steps for each scenario
            for step in range(4):
                batch = {
                    'data': f'scenario_{scenario_idx}_step_{step}',
                    'size': scenario['batch_size'],
                    'seq_len': scenario['seq_len']
                }
                
                step_start = time.time()
                
                try:
                    result = await trainer.train_step(batch)
                    step_end = time.time()
                    
                    completed_steps += 1
                    progress = (completed_steps / total_scenarios) * 100
                    
                    # Custom power-aware carbon calculation
                    step_duration_hours = (step_end - step_start) / 3600
                    power_kw = result['gpu_power_watts'] / 1000
                    step_energy = power_kw * step_duration_hours
                    
                    # Get current carbon intensity
                    current_intensity = await trainer.monitor.get_current_intensity('US-CA')
                    if current_intensity:
                        step_carbon = step_energy * (current_intensity.carbon_intensity / 1000)
                        
                        print(f"   ⚡ Step {step+1}: "
                              f"Loss {result['loss']:.3f}, "
                              f"Acc {result['accuracy']:.3f}, "
                              f"Power {result['gpu_power_watts']:.0f}W, "
                              f"Energy {step_energy*1000:.1f}Wh, "
                              f"Carbon {step_carbon*1000:.1f}g CO2 "
                              f"[{progress:.1f}%]")
                    
                    # Small delay to simulate realistic training pace
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    print(f"   ❌ Error in step {step}: {e}")
                    break
        
        # Find optimal training windows
        print("\n🔍 Analyzing Optimal Training Windows...")
        try:
            optimal_window = await trainer.find_optimal_training_window(
                duration_hours=4
            )
            
            if optimal_window:
                print(f"   📅 Optimal 4-hour window:")
                print(f"      Start: {optimal_window.start_time.strftime('%Y-%m-%d %H:%M')}")
                print(f"      End: {optimal_window.end_time.strftime('%Y-%m-%d %H:%M')}")
                print(f"      Avg Carbon: {optimal_window.avg_carbon_intensity:.1f} gCO2/kWh")
                print(f"      Renewables: {optimal_window.renewable_percentage:.1f}%")
                print(f"      Region: {optimal_window.region}")
        except Exception as e:
            print(f"   ⚠️  Window optimization unavailable: {e}")
        
        # Final comprehensive metrics
        final_metrics = trainer.get_carbon_metrics()
        
        print("\n" + "="*50)
        print("📊 COMPREHENSIVE TRAINING REPORT")
        print("="*50)
        
        print(f"🎯 Session: {final_metrics['session_id']}")
        print(f"⏰ Runtime: {final_metrics['runtime_hours']:.3f} hours")
        print(f"🔢 Steps: {trainer.step}")
        print(f"📈 FLOPs: {model.total_flops/1e12:.2f} TFLOPs")
        
        print(f"\n⚡ Energy & Carbon:")
        print(f"   Total Energy: {final_metrics['total_energy_kwh']:.4f} kWh")
        print(f"   Total Carbon: {final_metrics['total_carbon_kg']:.3f} kg CO2")
        print(f"   Avg Intensity: {final_metrics['avg_carbon_intensity']:.1f} gCO2/kWh")
        print(f"   Peak Intensity: {final_metrics['peak_carbon_intensity']:.1f} gCO2/kWh")
        print(f"   Min Intensity: {final_metrics['min_carbon_intensity']:.1f} gCO2/kWh")
        
        print(f"\n⏸️  Pausing:")
        print(f"   Paused Time: {final_metrics['paused_duration_hours']:.3f} hours")
        print(f"   Carbon Saved: {final_metrics['carbon_saved_kg']:.3f} kg CO2")
        
        # Calculate some interesting derived metrics
        if final_metrics['total_energy_kwh'] > 0:
            efficiency = model.total_flops / (final_metrics['total_energy_kwh'] * 1e12)
            print(f"   Compute Efficiency: {efficiency:.1f} TFLOP/kWh")
        
        if final_metrics['total_carbon_kg'] > 0:
            carbon_efficiency = model.total_flops / (final_metrics['total_carbon_kg'] * 1e12)
            print(f"   Carbon Efficiency: {carbon_efficiency:.1f} TFLOP/kg CO2")
    
    print("\n✅ Advanced carbon-aware training completed successfully!")
    print("🌱 Thank you for training responsibly!")


if __name__ == "__main__":
    asyncio.run(main())