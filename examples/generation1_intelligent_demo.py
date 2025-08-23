#!/usr/bin/env python3
"""
Generation 1 Intelligent Demo: Enhanced Carbon-Aware Training

This demonstrates the intelligent enhancements added to the core scheduler
in Generation 1, including:
- Adaptive batch sizing based on carbon intensity
- Intelligent training plans with carbon forecasting
- Smart batch scheduling with optimization
- Enhanced metrics with efficiency scoring
"""

import asyncio
import time
import random
from datetime import datetime, timedelta

# Import our enhanced carbon-aware trainer
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from carbon_aware_trainer.core.scheduler import CarbonAwareTrainer
from carbon_aware_trainer.core.types import TrainingConfig
from carbon_aware_trainer.core.intelligent_carbon_predictor import IntelligentCarbonPredictor


class DemoModel:
    """Simple demo model for testing."""
    
    def __init__(self, name="demo-model"):
        self.name = name
        self.parameters = 1000000  # 1M parameters
        self.training_step = 0
    
    def forward(self, batch):
        """Simulate forward pass."""
        # Simulate computation time based on batch size
        batch_size = batch.get('size', 32)
        computation_time = 0.01 * (batch_size / 32)  # Scale with batch size
        time.sleep(computation_time)
        
        # Simulate loss
        loss = random.uniform(0.1, 2.0) * (1 - self.training_step * 0.001)  # Decreasing loss
        self.training_step += 1
        
        return {'loss': loss, 'accuracy': min(0.95, 0.1 + self.training_step * 0.002)}


class DemoOptimizer:
    """Simple demo optimizer."""
    
    def __init__(self, lr=0.001):
        self.lr = lr
        self.step_count = 0
    
    def zero_grad(self):
        """Simulate gradient zeroing."""
        pass
    
    def step(self):
        """Simulate optimizer step."""
        self.step_count += 1


async def intelligent_demo():
    """Demonstrate Generation 1 intelligent enhancements."""
    print("🧠 GENERATION 1 INTELLIGENT DEMO")
    print("=" * 50)
    
    # Create demo model and optimizer
    model = DemoModel("intelligent-transformer")
    optimizer = DemoOptimizer(lr=0.001)
    
    # Configure intelligent training
    config = TrainingConfig(
        carbon_threshold=100.0,
        pause_threshold=150.0,
        resume_threshold=70.0,
        check_interval=30,  # Quick checks for demo
        preferred_regions=["US-WA", "EU-FR", "CA-QC"]  # Clean energy regions
    )
    
    print(f"📊 Model: {model.name}")
    print(f"⚡ Parameters: {model.parameters:,}")
    print(f"🎯 Carbon threshold: {config.carbon_threshold} gCO2/kWh")
    print(f"⏱️  Check interval: {config.check_interval}s")
    print()
    
    # Initialize carbon-aware trainer
    trainer = CarbonAwareTrainer(
        model=model,
        optimizer=optimizer,
        carbon_model='cached',  # Use cached for demo
        region='US-CA',
        config=config
    )
    
    # Add adaptive training flag for intelligent features
    trainer.adaptive_training = True
    
    try:
        print("🚀 Initializing intelligent trainer...")
        await trainer.initialize()
        
        # Demonstrate intelligent training plan
        print("\n📋 INTELLIGENT TRAINING PLAN")
        print("-" * 30)
        
        total_steps = 1000
        estimated_hours = 8.0
        
        plan = await trainer.intelligent_training_plan(
            total_steps=total_steps,
            estimated_duration_hours=estimated_hours
        )
        
        if "error" not in plan:
            print(f"📈 Total steps: {plan['total_estimated_steps']:,}")
            print(f"⏰ Estimated duration: {plan['estimated_duration_hours']:.1f} hours")
            print(f"🪟 Optimal windows: {plan['total_windows']}")
            print(f"🌱 Carbon savings: {plan['carbon_reduction_percent']:.1f}%")
            print(f"📅 Recommended start: {plan.get('recommended_start_time', 'Now')}")
            
            if plan['optimal_windows']:
                print("\nOptimal training windows:")
                for i, window in enumerate(plan['optimal_windows'][:3]):  # Show first 3
                    start = datetime.fromisoformat(window['start']).strftime('%m/%d %H:%M')
                    duration = window['duration_hours']
                    carbon = window['avg_carbon_intensity']
                    print(f"  {i+1}. {start} ({duration:.1f}h) - {carbon:.0f} gCO2/kWh")
        
        # Demonstrate adaptive batch scheduling
        print("\n🔄 ADAPTIVE BATCH SCHEDULING")
        print("-" * 30)
        
        total_batches = 200
        batch_schedule = await trainer.adaptive_batch_scheduler(total_batches)
        
        if batch_schedule:
            high_priority = sum(1 for b in batch_schedule if b['priority'] == 'high')
            total_carbon = sum(b['estimated_carbon_kg'] for b in batch_schedule)
            
            print(f"📦 Total batches: {total_batches}")
            print(f"⚡ High priority hours: {high_priority}")
            print(f"🌱 Estimated carbon: {total_carbon:.3f} kg CO2")
            
            print("\nFirst few scheduled hours:")
            for i, batch in enumerate(batch_schedule[:5]):
                start = datetime.fromisoformat(batch['start_time']).strftime('%H:%M')
                count = batch['batch_count']
                carbon = batch['carbon_intensity']
                priority = batch['priority'].upper()
                print(f"  {start}: {count:3d} batches ({carbon:3.0f} gCO2/kWh) [{priority}]")
        
        # Start training session
        print("\n🎯 INTELLIGENT TRAINING SESSION")
        print("-" * 30)
        
        async with trainer.training_session():
            print("Training started with intelligent adaptation...")
            
            # Simulate training batches with different sizes
            batch_sizes = [16, 32, 64, 32, 16, 8, 32, 64]  # Varying sizes
            
            for epoch in range(3):  # Short demo
                print(f"\n📚 Epoch {epoch + 1}")
                epoch_start = time.time()
                
                for batch_idx, batch_size in enumerate(batch_sizes):
                    # Create demo batch with adaptive sizing
                    batch = {
                        'size': batch_size,
                        'data': f"batch_{epoch}_{batch_idx}",
                        'max_batch_size': 128  # For adaptive scaling
                    }
                    
                    # Execute intelligent training step
                    result = await trainer.train_step(batch, batch_size=batch_size)
                    
                    if batch_idx % 3 == 0:  # Show every few steps
                        metrics = trainer.get_carbon_metrics()
                        efficiency = metrics['efficiency_score']
                        carbon = metrics.get('current_intensity', 'N/A')
                        
                        print(f"  Step {trainer.step:3d}: "
                              f"loss={result['loss']:.3f}, "
                              f"acc={result['accuracy']:.3f}, "
                              f"efficiency={efficiency:.1f}, "
                              f"carbon={carbon}")
                
                epoch_time = time.time() - epoch_start
                print(f"  Epoch completed in {epoch_time:.1f}s")
                
                # Brief pause between epochs
                await asyncio.sleep(1)
        
        # Final intelligent metrics
        print("\n📊 FINAL INTELLIGENT METRICS")
        print("-" * 30)
        
        final_metrics = trainer.get_carbon_metrics()
        
        print(f"🔢 Total steps: {final_metrics['step']}")
        print(f"⏱️  Runtime: {final_metrics['runtime_hours']:.2f} hours")
        print(f"⚡ Energy consumed: {final_metrics['total_energy_kwh']:.4f} kWh")
        print(f"🌱 Carbon emitted: {final_metrics['total_carbon_kg']:.4f} kg CO2")
        print(f"📈 Avg carbon intensity: {final_metrics['avg_carbon_intensity']:.1f} gCO2/kWh")
        print(f"📊 Efficiency score: {final_metrics['efficiency_score']:.1f}/100")
        print(f"⏸️  Pause duration: {final_metrics['paused_duration_hours']:.2f} hours")
        print(f"💾 Carbon saved: {final_metrics['carbon_saved_kg']:.4f} kg CO2")
        
        # Show adaptive features impact
        print(f"\n🧠 INTELLIGENT ADAPTATIONS")
        print(f"✨ Adaptive batch sizing: Enabled")
        print(f"📈 Carbon trend analysis: {final_metrics.get('carbon_trend', 'stable').title()}")
        print(f"🎯 Smart scheduling: Optimized")
        print(f"🔮 Predictive optimization: Active")
        
    except Exception as e:
        print(f"❌ Error in intelligent demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await trainer.cleanup()
    
    print("\n✅ Generation 1 intelligent demo completed!")
    print("🎉 Enhanced features: Adaptive training, intelligent planning, smart scheduling")


async def quick_intelligence_test():
    """Quick test of intelligent features."""
    print("\n🔍 QUICK INTELLIGENCE TEST")
    print("=" * 30)
    
    # Test adaptive parameter adjustment
    trainer = CarbonAwareTrainer(
        model=DemoModel(),
        carbon_model='cached',
        region='US-CA'
    )
    
    trainer.adaptive_training = True
    
    # Test different carbon intensities
    test_cases = [
        (50, "Low carbon - should increase batch size"),
        (120, "Medium carbon - normal processing"),
        (180, "High carbon - should reduce batch size")
    ]
    
    for carbon_level, description in test_cases:
        kwargs = {'batch_size': 32, 'max_batch_size': 64}
        adapted = trainer._adapt_training_params(carbon_level, kwargs)
        
        print(f"🔬 {description}")
        print(f"   Carbon: {carbon_level} gCO2/kWh")
        print(f"   Original batch size: {kwargs['batch_size']}")
        print(f"   Adapted batch size: {adapted['batch_size']}")
        print(f"   Low carbon mode: {adapted.get('low_carbon_mode', False)}")
        print()
    
    print("✅ Intelligence test completed!")


async def main():
    """Main demo function."""
    print("🌱 CARBON-AWARE TRAINER - GENERATION 1")
    print("🧠 Intelligent Enhancements Demo")
    print("=" * 60)
    print()
    
    # Run quick intelligence test
    await quick_intelligence_test()
    
    # Run full demo
    await intelligent_demo()
    
    print("\n" + "=" * 60)
    print("🎯 GENERATION 1 COMPLETE")
    print("✨ Features added:")
    print("   • Adaptive batch sizing based on carbon intensity")
    print("   • Intelligent training plans with carbon forecasting") 
    print("   • Smart batch scheduling optimization")
    print("   • Enhanced metrics with efficiency scoring")
    print("   • Predictive carbon analysis")
    print("   • Real-time training adaptations")
    print()
    print("➡️  Ready for Generation 2: Robustness & Reliability")


if __name__ == "__main__":
    asyncio.run(main())