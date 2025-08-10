#!/usr/bin/env python3
"""Advanced auto-scaling demonstration with multi-objective optimization."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, timedelta
import random
import logging

from carbon_aware_trainer import CarbonAwareTrainer
from carbon_aware_trainer.core.types import TrainingConfig, TrainingState
from carbon_aware_trainer.core.auto_scaling import (
    AutoScalingOptimizer, ScalingStrategy, ResourceType, ResourceMetrics
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdaptiveTrainingModel:
    """Advanced model with dynamic resource adaptation."""
    
    def __init__(self):
        self.step_count = 0
        self.batch_size = 32
        self.learning_rate = 0.001
        self.parallelism = 1
        self.gpu_devices = 1
        
        # Performance simulation
        self.base_throughput = 100.0  # operations per second
        self.base_gpu_utilization = 0.7
        self.base_memory_usage = 0.6
        
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource utilization metrics."""
        # Simulate resource usage based on current configuration
        
        # Batch size affects utilization
        batch_factor = self.batch_size / 32.0
        gpu_util = min(0.98, self.base_gpu_utilization * batch_factor)
        memory_util = min(0.95, self.base_memory_usage * batch_factor)
        
        # Add some randomness for realism
        gpu_util += random.uniform(-0.1, 0.1)
        memory_util += random.uniform(-0.05, 0.05)
        
        # Clamp values
        gpu_util = max(0.1, min(0.98, gpu_util))
        memory_util = max(0.1, min(0.95, memory_util))
        
        # Throughput based on parallelism and batch size
        throughput = self.base_throughput * self.parallelism * (self.batch_size / 32.0) ** 0.8
        
        # Carbon efficiency (operations per gram CO2)
        # Higher efficiency with better utilization
        carbon_efficiency = throughput / max(0.1, gpu_util) * 0.1
        
        # Cost efficiency (operations per dollar)
        cost_efficiency = throughput / (gpu_util * self.gpu_devices * 0.5)
        
        return ResourceMetrics(
            cpu_utilization=0.4 + random.uniform(-0.1, 0.1),
            gpu_utilization=gpu_util,
            memory_utilization=memory_util,
            network_bandwidth=random.uniform(0.1, 0.5),
            throughput_ops_per_sec=throughput,
            latency_ms=1000.0 / throughput,
            carbon_efficiency=carbon_efficiency,
            cost_efficiency=cost_efficiency
        )
    
    def apply_scaling_decision(self, resource_type: ResourceType, new_value: float) -> bool:
        """Apply a scaling decision to the model."""
        try:
            if resource_type == ResourceType.BATCH_SIZE:
                old_value = self.batch_size
                self.batch_size = int(max(1, min(512, new_value)))
                print(f"   📏 Batch size: {old_value} → {self.batch_size}")
                
            elif resource_type == ResourceType.LEARNING_RATE:
                old_value = self.learning_rate
                self.learning_rate = max(1e-6, min(1e-1, new_value))
                print(f"   📈 Learning rate: {old_value:.2e} → {self.learning_rate:.2e}")
                
            elif resource_type == ResourceType.PARALLELISM:
                old_value = self.parallelism
                self.parallelism = int(max(1, min(16, new_value)))
                print(f"   ⚡ Parallelism: {old_value} → {self.parallelism}")
                
            elif resource_type == ResourceType.GPU_DEVICES:
                old_value = self.gpu_devices
                self.gpu_devices = int(max(1, min(8, new_value)))
                print(f"   🖥️  GPU devices: {old_value} → {self.gpu_devices}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply scaling decision: {e}")
            return False
    
    def train_step(self, batch, **kwargs):
        """Execute training step with current configuration."""
        self.step_count += 1
        
        # Performance varies with batch size and parallelism
        step_time = 1.0 / (self.batch_size * self.parallelism / 100.0)
        energy_per_step = 0.001 * self.batch_size * self.gpu_devices
        
        return {
            'loss': max(0.01, 2.0 - (self.step_count * 0.02)),
            'accuracy': min(0.98, self.step_count * 0.015),
            'step_time': step_time,
            'energy_kwh': energy_per_step,
            'batch_size': self.batch_size,
            'throughput': self.batch_size / step_time
        }


async def demonstrate_auto_scaling():
    """Demonstrate advanced auto-scaling optimization."""
    print("🚀 Advanced Auto-Scaling Optimization Demo")
    print("=" * 50)
    
    # Initialize different optimizers for comparison
    optimizers = {
        'carbon_aware': AutoScalingOptimizer(
            strategy=ScalingStrategy.CARBON_AWARE,
            min_scaling_interval=10,  # 10 seconds for demo
            carbon_weight=0.7,
            performance_weight=0.2,
            cost_weight=0.1
        ),
        'performance_focused': AutoScalingOptimizer(
            strategy=ScalingStrategy.PERFORMANCE_BASED,
            min_scaling_interval=10,
            carbon_weight=0.1,
            performance_weight=0.8,
            cost_weight=0.1
        ),
        'hybrid': AutoScalingOptimizer(
            strategy=ScalingStrategy.HYBRID,
            min_scaling_interval=10,
            carbon_weight=0.4,
            performance_weight=0.4,
            cost_weight=0.2
        )
    }
    
    # Test each optimizer
    for optimizer_name, optimizer in optimizers.items():
        print(f"\n🧪 Testing {optimizer_name.upper()} Optimization Strategy")
        print("-" * 40)
        
        # Create adaptive model
        model = AdaptiveTrainingModel()
        
        # Training configuration
        config = TrainingConfig(
            carbon_threshold=120.0,
            pause_threshold=200.0,
            resume_threshold=80.0,
            check_interval=5
        )
        
        # Initialize trainer
        trainer = CarbonAwareTrainer(
            model=model,
            carbon_model='cached',
            region='US-CA',
            config=config,
            api_key='sample_data/sample_carbon_data.json'
        )
        
        # Simulate different carbon intensity scenarios
        carbon_scenarios = [
            {'intensity': 80, 'description': 'Low carbon (clean energy)'},
            {'intensity': 150, 'description': 'Medium carbon (mixed grid)'},
            {'intensity': 300, 'description': 'High carbon (fossil heavy)'},
            {'intensity': 120, 'description': 'Normal carbon (balanced)'}
        ]
        
        total_steps = 0
        total_optimizations = 0
        performance_history = []
        
        async with trainer:
            await trainer.start_training()
            
            for scenario_idx, scenario in enumerate(carbon_scenarios):
                print(f"\n📊 Scenario {scenario_idx + 1}: {scenario['description']} "
                      f"({scenario['intensity']} gCO2/kWh)")
                
                # Simulate carbon intensity
                from carbon_aware_trainer.core.types import CarbonIntensity
                carbon_intensity = CarbonIntensity(
                    carbon_intensity=scenario['intensity'],
                    timestamp=datetime.now(),
                    region='US-CA'
                )
                
                # Run optimization cycles for this scenario
                for cycle in range(5):  # 5 optimization cycles per scenario
                    current_metrics = model.get_current_metrics()
                    
                    print(f"\n   🔍 Optimization Cycle {cycle + 1}")
                    print(f"      GPU Util: {current_metrics.gpu_utilization:.1%}")
                    print(f"      Memory: {current_metrics.memory_utilization:.1%}")
                    print(f"      Throughput: {current_metrics.throughput_ops_per_sec:.1f} ops/s")
                    print(f"      Carbon Efficiency: {current_metrics.carbon_efficiency:.2f} ops/g CO2")
                    
                    # Get scaling decisions
                    decisions = await optimizer.analyze_and_optimize(
                        current_metrics=current_metrics,
                        carbon_intensity=carbon_intensity,
                        training_state=TrainingState.RUNNING
                    )
                    
                    # Apply scaling decisions
                    applied_changes = 0
                    for decision in decisions:
                        print(f"   🎯 {decision.action.upper()}: {decision.resource_type.value}")
                        print(f"      {decision.current_value:.2f} → {decision.target_value:.2f}")
                        print(f"      Confidence: {decision.confidence:.1%}")
                        print(f"      Rationale: {decision.rationale}")
                        
                        # Apply the change
                        success = model.apply_scaling_decision(
                            decision.resource_type, 
                            decision.target_value
                        )
                        
                        if success:
                            applied_changes += 1
                            
                            # Simulate benefits
                            benefit = {}
                            if decision.action == 'scale_up':
                                benefit['performance_improvement'] = 0.1
                            if scenario['intensity'] > 200:
                                benefit['carbon_savings_kg'] = 0.05
                            
                            optimizer.complete_scaling_change(
                                decision.resource_type, 
                                True, 
                                benefit
                            )
                    
                    total_optimizations += applied_changes
                    
                    # Run a few training steps to see the effect
                    step_results = []
                    for step in range(3):
                        batch = {'data': f'batch_{total_steps}', 'size': model.batch_size}
                        result = await trainer.train_step(batch)
                        step_results.append(result)
                        total_steps += 1
                    
                    # Calculate average performance
                    avg_throughput = sum(r['throughput'] for r in step_results) / len(step_results)
                    performance_history.append({
                        'cycle': len(performance_history),
                        'scenario': scenario['description'],
                        'throughput': avg_throughput,
                        'batch_size': model.batch_size,
                        'carbon_intensity': scenario['intensity']
                    })
                    
                    await asyncio.sleep(0.5)  # Small delay between cycles
        
        # Strategy performance summary
        optimizer_summary = optimizer.get_optimization_summary()
        
        print(f"\n📊 {optimizer_name.upper()} Strategy Summary:")
        print(f"   Total Decisions: {optimizer_summary['total_decisions']}")
        print(f"   Successful Optimizations: {optimizer_summary['successful_optimizations']}")
        print(f"   Success Rate: {optimizer_summary['success_rate']:.1%}")
        print(f"   Carbon Savings: {optimizer_summary['carbon_savings_kg']:.3f} kg CO2")
        print(f"   Performance Improvements: {optimizer_summary['performance_improvements']:.2f}")
        print(f"   Cost Savings: ${optimizer_summary['cost_savings_usd']:.2f}")
        
        # Performance analysis
        if performance_history:
            avg_throughput = sum(p['throughput'] for p in performance_history) / len(performance_history)
            max_throughput = max(p['throughput'] for p in performance_history)
            final_throughput = performance_history[-1]['throughput']
            
            print(f"\n📈 Performance Analysis:")
            print(f"   Average Throughput: {avg_throughput:.1f} ops/s")
            print(f"   Peak Throughput: {max_throughput:.1f} ops/s")
            print(f"   Final Throughput: {final_throughput:.1f} ops/s")
            print(f"   Performance Improvement: {((final_throughput / performance_history[0]['throughput']) - 1) * 100:+.1f}%")
        
        print("\n" + "=" * 50)
    
    print("\n🎉 Auto-Scaling Optimization Demo Completed!")
    print("🔬 Demonstrated multi-objective optimization with:")
    print("   ✅ Carbon-aware resource scaling")
    print("   ✅ Performance-based optimization")  
    print("   ✅ Hybrid multi-objective strategies")
    print("   ✅ Adaptive batch sizing")
    print("   ✅ Real-time decision making")
    print("   ✅ Multi-scenario adaptability")


if __name__ == "__main__":
    asyncio.run(demonstrate_auto_scaling())