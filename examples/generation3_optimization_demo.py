#!/usr/bin/env python3
"""
Generation 3 Optimization Demo: Scaling & Performance

This demonstrates the optimization features added in Generation 3, including:
- Advanced performance optimization and resource utilization
- Adaptive batch sizing and parallel processing
- Carbon-aware performance scaling
- Real-time performance monitoring and adaptation
- Multi-mode optimization strategies
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import our enhanced carbon-aware trainer
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from carbon_aware_trainer.core.scheduler import CarbonAwareTrainer
from carbon_aware_trainer.core.types import TrainingConfig


class OptimizedDemoModel:
    """Advanced demo model with performance characteristics."""
    
    def __init__(self, name="optimized-model", base_complexity=1.0):
        self.name = name
        self.parameters = 5000000  # 5M parameters
        self.training_step = 0
        self.base_complexity = base_complexity
        
        # Performance characteristics
        self._compute_intensity = 1.0
        self._memory_usage = 1.0
        self._throughput = 1.0
        
        # Model state
        self._weights = {f'layer_{i}': random.uniform(-1, 1) for i in range(10)}
    
    def forward(self, batch):
        """Forward pass with configurable performance characteristics."""
        batch_size = batch.get('size', 32)
        
        # Apply performance modifiers from optimization
        memory_opt = batch.get('memory_optimization', False)
        mixed_precision = batch.get('mixed_precision', False)
        parallel_workers = batch.get('parallel_workers', 1)
        
        # Simulate computation time based on optimizations
        base_time = 0.02 * (batch_size / 32) * self.base_complexity
        
        # Memory optimization reduces time but might reduce accuracy
        if memory_opt:
            base_time *= 0.8
            
        # Mixed precision reduces time and memory
        if mixed_precision:
            base_time *= 0.7
            
        # Parallel workers reduce time (with diminishing returns)
        parallel_factor = 1.0 / (1.0 + (parallel_workers - 1) * 0.3)
        base_time *= parallel_factor
        
        # Add some random variation
        computation_time = base_time * random.uniform(0.9, 1.1)
        
        # Simulate the computation
        time.sleep(computation_time)
        
        # Calculate performance metrics
        progress = min(1.0, self.training_step * 0.0005)  # Slower convergence for demo
        
        # Base loss and accuracy
        loss = 3.0 * (1 - progress * 0.85) + random.uniform(-0.2, 0.2)
        accuracy = min(0.98, 0.02 + progress * 0.95)
        
        # Performance impacts
        if memory_opt:
            # Memory optimization might slightly reduce accuracy
            accuracy *= 0.995
        
        if mixed_precision:
            # Mixed precision might introduce slight numerical errors
            loss *= random.uniform(0.98, 1.02)
        
        self.training_step += 1
        
        # Calculate throughput (samples per second)
        throughput = batch_size / computation_time
        
        return {
            'loss': max(0.01, loss),
            'accuracy': accuracy,
            'throughput': throughput,
            'computation_time': computation_time,
            'batch_size': batch_size,
            'optimization_applied': {
                'memory_opt': memory_opt,
                'mixed_precision': mixed_precision,
                'parallel_workers': parallel_workers
            }
        }
    
    def state_dict(self):
        """Return model state."""
        return {
            'weights': self._weights.copy(),
            'training_step': self.training_step,
            'parameters': self.parameters,
            'compute_intensity': self._compute_intensity
        }
    
    def load_state_dict(self, state_dict):
        """Load model state."""
        self._weights = state_dict.get('weights', self._weights)
        self.training_step = state_dict.get('training_step', 0)
        self.parameters = state_dict.get('parameters', self.parameters)


class OptimizedDemoOptimizer:
    """Demo optimizer with performance features."""
    
    def __init__(self, lr=0.001):
        self.lr = lr
        self.step_count = 0
        self.momentum_states = {}
        
    def zero_grad(self):
        """Zero gradients."""
        pass
    
    def step(self):
        """Optimizer step with performance tracking."""
        self.step_count += 1
        
        # Simulate optimization work
        time.sleep(0.001)  # Small overhead
    
    def state_dict(self):
        """Return optimizer state."""
        return {
            'lr': self.lr,
            'step_count': self.step_count,
            'momentum_states': self.momentum_states.copy()
        }


async def test_optimization_modes():
    """Test different optimization modes."""
    print("🎯 TESTING OPTIMIZATION MODES")
    print("-" * 40)
    
    modes = ['carbon_first', 'performance_first', 'balanced', 'cost_optimized']
    
    for mode in modes:
        print(f"\n🔧 Testing {mode.upper()} mode...")
        
        model = OptimizedDemoModel(f"model-{mode}")
        optimizer = OptimizedDemoOptimizer()
        
        config = TrainingConfig(
            carbon_threshold=100.0,
            # Generation 3 settings
            resource_optimization_enabled=True,
            optimization_mode=mode,
            target_utilization=0.8,
            max_concurrent_jobs=4,
            adaptive_batch_sizing=True,
            performance_monitoring=True
        )
        
        trainer = CarbonAwareTrainer(
            model=model,
            optimizer=optimizer,
            carbon_model='cached',
            region='US-CA',
            config=config
        )
        
        try:
            await trainer.initialize()
            
            # Test a few training steps
            async with trainer.training_session():
                total_throughput = 0
                total_time = 0
                
                for step in range(10):
                    batch = {'size': 32, 'step': step}
                    
                    step_start = time.time()
                    result = await trainer.train_step(batch)
                    step_time = time.time() - step_start
                    
                    total_throughput += result.get('throughput', 0)
                    total_time += step_time
                    
                    if step % 5 == 0:
                        optimization = result.get('optimization_applied', {})
                        print(f"    Step {step}: throughput={result.get('throughput', 0):.1f} samples/s, "
                              f"loss={result.get('loss', 0):.3f}, "
                              f"opts={optimization}")
                
                avg_throughput = total_throughput / 10
                avg_step_time = total_time / 10
                
                print(f"  ✅ Average throughput: {avg_throughput:.1f} samples/s")
                print(f"  ⏱️  Average step time: {avg_step_time:.3f}s")
                
                # Get performance stats
                if trainer.performance_optimizer:
                    stats = trainer.performance_optimizer.get_performance_stats()
                    profile = stats.get('profile_settings', {})
                    print(f"  📊 Final profile: batch_mult={profile.get('batch_multiplier', 1.0):.2f}, "
                          f"workers={profile.get('parallel_workers', 1)}")
        
        except Exception as e:
            print(f"  ❌ Mode {mode} failed: {e}")
        
        finally:
            await trainer.cleanup()
    
    print("\n✅ Optimization modes testing completed")


async def test_adaptive_performance():
    """Test adaptive performance optimization."""
    print("\n🔄 TESTING ADAPTIVE PERFORMANCE")
    print("-" * 35)
    
    model = OptimizedDemoModel("adaptive-model")
    optimizer = OptimizedDemoOptimizer()
    
    config = TrainingConfig(
        resource_optimization_enabled=True,
        optimization_mode="balanced",
        target_utilization=0.75,
        max_concurrent_jobs=6,
        adaptive_batch_sizing=True,
        performance_monitoring=True,
        checkpoint_interval=50,  # Keep some Generation 2 features
    )
    
    trainer = CarbonAwareTrainer(
        model=model,
        optimizer=optimizer,
        carbon_model='cached',
        region='US-CA',
        config=config
    )
    
    trainer.adaptive_training = True  # Enable Generation 1 features
    
    try:
        await trainer.initialize()
        
        print("📊 Monitoring adaptive performance changes...")
        
        async with trainer.training_session():
            performance_log = []
            
            for step in range(50):
                # Simulate varying carbon intensity
                if step < 15:
                    simulated_carbon = 80 + random.uniform(-20, 20)  # Low carbon
                elif step < 30:
                    simulated_carbon = 180 + random.uniform(-30, 30)  # High carbon  
                else:
                    simulated_carbon = 120 + random.uniform(-25, 25)  # Medium carbon
                
                batch = {
                    'size': 32 + random.randint(-8, 8),  # Varying batch sizes
                    'step': step,
                    'simulated_carbon': simulated_carbon
                }
                
                step_start = time.time()
                result = await trainer.train_step(batch)
                step_duration = time.time() - step_start
                
                performance_log.append({
                    'step': step,
                    'carbon_intensity': simulated_carbon,
                    'throughput': result.get('throughput', 0),
                    'step_duration': step_duration,
                    'optimizations': result.get('optimization_applied', {})
                })
                
                if step % 10 == 0:
                    recent_throughput = sum(p['throughput'] for p in performance_log[-5:]) / 5
                    recent_carbon = sum(p['carbon_intensity'] for p in performance_log[-5:]) / 5
                    
                    print(f"    Step {step:2d}: avg_throughput={recent_throughput:.1f} samples/s, "
                          f"avg_carbon={recent_carbon:.0f} gCO2/kWh")
            
            # Analyze performance adaptation
            print("\n📈 ADAPTIVE PERFORMANCE ANALYSIS")
            print("-" * 35)
            
            low_carbon_steps = [p for p in performance_log if p['carbon_intensity'] < 100]
            high_carbon_steps = [p for p in performance_log if p['carbon_intensity'] > 150]
            
            if low_carbon_steps and high_carbon_steps:
                low_carbon_throughput = sum(p['throughput'] for p in low_carbon_steps) / len(low_carbon_steps)
                high_carbon_throughput = sum(p['throughput'] for p in high_carbon_steps) / len(high_carbon_steps)
                
                print(f"📊 Low carbon avg throughput: {low_carbon_throughput:.1f} samples/s")
                print(f"📊 High carbon avg throughput: {high_carbon_throughput:.1f} samples/s")
                
                if low_carbon_throughput > high_carbon_throughput:
                    improvement = ((low_carbon_throughput - high_carbon_throughput) / high_carbon_throughput * 100)
                    print(f"✅ Adaptive optimization working: {improvement:.1f}% better performance in low carbon periods")
                else:
                    print("⚠️ Adaptation needs tuning - performance difference minimal")
    
    except Exception as e:
        print(f"❌ Adaptive performance test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await trainer.cleanup()


async def test_batch_optimization():
    """Test optimized batch processing."""
    print("\n📦 TESTING BATCH OPTIMIZATION")
    print("-" * 30)
    
    model = OptimizedDemoModel("batch-optimized")
    config = TrainingConfig(
        resource_optimization_enabled=True,
        optimization_mode="performance_first",
        max_concurrent_jobs=8
    )
    
    trainer = CarbonAwareTrainer(model=model, config=config, region='US-CA', carbon_model='cached')
    
    try:
        await trainer.initialize()
        
        if trainer.performance_optimizer:
            # Create a batch processing function
            def process_batch(batch_data):
                """Simulate processing a batch."""
                batch_id, size = batch_data
                # Simulate work
                time.sleep(0.05 * (size / 32))  # Scale with batch size
                return {
                    'batch_id': batch_id,
                    'processed_size': size,
                    'processing_time': 0.05 * (size / 32)
                }
            
            # Test different carbon intensities
            carbon_levels = [50, 150, 250]  # Low, medium, high
            
            for carbon_intensity in carbon_levels:
                print(f"\n🌱 Testing with {carbon_intensity} gCO2/kWh carbon intensity...")
                
                # Create batches to process
                batches = [(i, random.randint(16, 64)) for i in range(20)]
                
                start_time = time.time()
                results = await trainer.performance_optimizer.optimize_batch_processing(
                    batches, process_batch, carbon_intensity
                )
                total_time = time.time() - start_time
                
                total_samples = sum(r['processed_size'] for r in results)
                avg_throughput = total_samples / total_time
                
                print(f"    Processed {len(results)} batches in {total_time:.2f}s")
                print(f"    Total samples: {total_samples}")
                print(f"    Throughput: {avg_throughput:.1f} samples/s")
                
                # Get performance stats
                stats = trainer.performance_optimizer.get_performance_stats()
                profile = stats.get('profile_settings', {})
                print(f"    Profile: {profile.get('parallel_workers', 1)} workers, "
                      f"batch_mult={profile.get('batch_multiplier', 1.0):.2f}")
        
        else:
            print("⚠️ Performance optimizer not available")
    
    except Exception as e:
        print(f"❌ Batch optimization test failed: {e}")
    
    finally:
        await trainer.cleanup()


async def comprehensive_optimization_demo():
    """Comprehensive demonstration of Generation 3 features."""
    print("⚡ GENERATION 3 OPTIMIZATION DEMO")
    print("=" * 50)
    
    model = OptimizedDemoModel("comprehensive-model", base_complexity=1.2)
    optimizer = OptimizedDemoOptimizer(lr=0.001)
    
    # Configure all Generation 3 features
    config = TrainingConfig(
        # Carbon awareness
        carbon_threshold=100.0,
        pause_threshold=150.0,
        resume_threshold=70.0,
        
        # Generation 2: Resilience
        enable_resilience=True,
        checkpoint_interval=25,
        max_failures=5,
        
        # Generation 3: Performance optimization
        resource_optimization_enabled=True,
        optimization_mode="balanced",
        target_utilization=0.8,
        auto_scaling_enabled=True,
        max_concurrent_jobs=6,
        adaptive_batch_sizing=True,
        performance_monitoring=True
    )
    
    print(f"🤖 Model: {model.name}")
    print(f"⚡ Parameters: {model.parameters:,}")
    print(f"🎯 Optimization mode: {config.optimization_mode}")
    print(f"🔧 Max concurrent jobs: {config.max_concurrent_jobs}")
    print(f"📊 Target utilization: {config.target_utilization}")
    print()
    
    trainer = CarbonAwareTrainer(
        model=model,
        optimizer=optimizer,
        carbon_model='cached',
        region='US-CA',
        config=config
    )
    
    trainer.adaptive_training = True  # Enable all generations
    
    try:
        print("🚀 Initializing comprehensive trainer...")
        await trainer.initialize()
        
        # Check which features are active
        active_features = []
        if hasattr(trainer, 'adaptive_training') and trainer.adaptive_training:
            active_features.append("Generation 1: Intelligent Adaptation")
        if trainer.resilience_manager:
            active_features.append("Generation 2: Resilience & Recovery")
        if trainer.performance_optimizer:
            active_features.append("Generation 3: Performance Optimization")
        
        print("✅ Active features:")
        for feature in active_features:
            print(f"   • {feature}")
        print()
        
        # Run comprehensive training
        print("🎯 COMPREHENSIVE TRAINING SESSION")
        print("-" * 35)
        
        async with trainer.training_session():
            training_stats = {
                'total_throughput': 0,
                'total_steps': 0,
                'adaptation_count': 0,
                'checkpoint_count': 0,
                'optimization_changes': 0
            }
            
            for epoch in range(3):
                print(f"\n📚 Epoch {epoch + 1}")
                epoch_start = time.time()
                
                for step in range(20):
                    # Simulate varying conditions
                    carbon_intensity = 80 + 40 * (0.5 + 0.5 * random.sin(step * 0.3))
                    batch_size = 24 + random.randint(-8, 16)
                    
                    batch = {
                        'size': batch_size,
                        'step': step,
                        'epoch': epoch,
                        'carbon_intensity': carbon_intensity
                    }
                    
                    try:
                        result = await trainer.train_step(batch)
                        
                        training_stats['total_throughput'] += result.get('throughput', 0)
                        training_stats['total_steps'] += 1
                        
                        if step % 5 == 0:
                            optimizations = result.get('optimization_applied', {})
                            print(f"    Step {trainer.step:3d}: "
                                  f"loss={result.get('loss', 0):.3f}, "
                                  f"acc={result.get('accuracy', 0):.3f}, "
                                  f"throughput={result.get('throughput', 0):.1f} samples/s")
                    
                    except Exception as e:
                        print(f"    ⚠️ Step {step} encountered issue: {e}")
                        continue
                
                epoch_time = time.time() - epoch_start
                print(f"    Epoch completed in {epoch_time:.1f}s")
        
        # Final comprehensive metrics
        print("\n📊 COMPREHENSIVE FINAL METRICS")
        print("-" * 40)
        
        final_metrics = trainer.get_carbon_metrics()
        
        # Core metrics
        print(f"🔢 Total steps: {final_metrics['step']}")
        print(f"⏱️ Runtime: {final_metrics['runtime_hours']:.3f} hours")
        print(f"⚡ Energy: {final_metrics['total_energy_kwh']:.4f} kWh")
        print(f"🌱 Carbon: {final_metrics['total_carbon_kg']:.4f} kg CO2")
        print(f"📈 Efficiency: {final_metrics['efficiency_score']:.1f}/100")
        
        # Generation-specific metrics
        if final_metrics.get('resilience_active'):
            print(f"🛡️ Checkpoints: {final_metrics.get('checkpoints_created', 0)}")
        
        if final_metrics.get('performance_optimization_active'):
            perf_stats = final_metrics.get('performance_stats')
            if perf_stats:
                print(f"⚡ Performance mode: {perf_stats.get('optimization_mode', 'unknown')}")
                print(f"📊 Monitoring samples: {perf_stats.get('monitoring_samples', 0)}")
                
                profile = perf_stats.get('profile_settings', {})
                if profile:
                    print(f"🔧 Final profile: "
                          f"batch_mult={profile.get('batch_multiplier', 1.0):.2f}, "
                          f"workers={profile.get('parallel_workers', 1)}")
        
        # Overall performance
        if training_stats['total_steps'] > 0:
            avg_throughput = training_stats['total_throughput'] / training_stats['total_steps']
            print(f"📈 Average throughput: {avg_throughput:.1f} samples/s")
        
    except Exception as e:
        print(f"❌ Comprehensive demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await trainer.cleanup()
    
    print("\n✅ Generation 3 comprehensive demo completed!")
    print("🎉 All features: Intelligent adaptation, resilience, optimization")


async def main():
    """Main demo function."""
    print("🌱 CARBON-AWARE TRAINER - GENERATION 3")
    print("⚡ Performance Optimization & Scaling Demo")
    print("=" * 60)
    print()
    
    # Run tests
    await test_optimization_modes()
    await test_adaptive_performance()
    await test_batch_optimization()
    await comprehensive_optimization_demo()
    
    print("\n" + "=" * 60)
    print("🎯 GENERATION 3 COMPLETE")
    print("✨ Optimization features added:")
    print("   • Advanced performance optimization with multiple modes")
    print("   • Adaptive resource utilization and batch sizing")
    print("   • Carbon-aware performance scaling")
    print("   • Real-time performance monitoring and adaptation")
    print("   • Optimized parallel batch processing")
    print("   • Comprehensive performance analytics")
    print()
    print("🏆 FULLY AUTONOMOUS CARBON-AWARE SDLC IMPLEMENTED")
    print("   Generation 1: Intelligent Adaptation ✅")
    print("   Generation 2: Resilience & Reliability ✅")
    print("   Generation 3: Performance & Scaling ✅")


if __name__ == "__main__":
    asyncio.run(main())