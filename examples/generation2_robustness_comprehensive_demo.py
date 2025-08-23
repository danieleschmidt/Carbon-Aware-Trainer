#!/usr/bin/env python3
"""
Generation 2 Robustness Demo: Comprehensive Resilience & Reliability

This demonstrates the robustness features added in Generation 2, including:
- Automatic checkpointing and recovery
- Comprehensive error handling and retry logic
- Circuit breaker patterns for external services
- Health monitoring and failure recovery
- Production-ready resilience management
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


class RobustDemoModel:
    """Demo model with failure simulation for testing robustness."""
    
    def __init__(self, name="robust-model", failure_rate=0.05):
        self.name = name
        self.parameters = 2000000  # 2M parameters
        self.training_step = 0
        self.failure_rate = failure_rate  # Probability of failure per step
        
        # Simulate internal state
        self._weights = {'layer1': 0.5, 'layer2': -0.3, 'layer3': 0.8}
    
    def forward(self, batch):
        """Simulate forward pass with potential failures."""
        # Simulate occasional failures
        if random.random() < self.failure_rate:
            if random.random() < 0.3:
                raise RuntimeError(f"Simulated compute failure at step {self.training_step}")
            elif random.random() < 0.5:
                raise MemoryError(f"Simulated OOM at step {self.training_step}")
            else:
                raise ConnectionError(f"Simulated network failure at step {self.training_step}")
        
        # Simulate computation time based on batch size
        batch_size = batch.get('size', 32)
        computation_time = 0.02 * (batch_size / 32)  # Scale with batch size
        time.sleep(computation_time)
        
        # Simulate loss that improves over time
        progress = min(1.0, self.training_step * 0.001)
        base_loss = 2.5 * (1 - progress * 0.8)  # Decreasing loss
        noise = random.uniform(-0.1, 0.1)
        loss = max(0.01, base_loss + noise)
        
        # Simulate accuracy
        accuracy = min(0.98, 0.05 + progress * 0.9)
        
        self.training_step += 1
        
        return {
            'loss': loss, 
            'accuracy': accuracy,
            'step': self.training_step
        }
    
    def state_dict(self):
        """Return model state for checkpointing."""
        return {
            'weights': self._weights.copy(),
            'training_step': self.training_step,
            'parameters': self.parameters
        }
    
    def load_state_dict(self, state_dict):
        """Load model state from checkpoint."""
        self._weights = state_dict.get('weights', self._weights)
        self.training_step = state_dict.get('training_step', 0)
        self.parameters = state_dict.get('parameters', self.parameters)


class RobustDemoOptimizer:
    """Demo optimizer with state persistence."""
    
    def __init__(self, lr=0.001):
        self.lr = lr
        self.step_count = 0
        self.momentum = {'layer1': 0.0, 'layer2': 0.0, 'layer3': 0.0}
    
    def zero_grad(self):
        """Simulate gradient zeroing."""
        pass
    
    def step(self):
        """Simulate optimizer step."""
        self.step_count += 1
        # Update momentum (simplified)
        for key in self.momentum:
            self.momentum[key] *= 0.9
    
    def state_dict(self):
        """Return optimizer state for checkpointing."""
        return {
            'lr': self.lr,
            'step_count': self.step_count,
            'momentum': self.momentum.copy()
        }
    
    def load_state_dict(self, state_dict):
        """Load optimizer state from checkpoint."""
        self.lr = state_dict.get('lr', self.lr)
        self.step_count = state_dict.get('step_count', 0)
        self.momentum = state_dict.get('momentum', self.momentum)


async def test_automatic_checkpointing(trainer, model, optimizer):
    """Test automatic checkpointing functionality."""
    print("🔄 TESTING AUTOMATIC CHECKPOINTING")
    print("-" * 40)
    
    checkpoint_interval = 25  # Checkpoint every 25 steps
    total_steps = 100
    
    print(f"📦 Checkpoint interval: {checkpoint_interval} steps")
    print(f"🎯 Target steps: {total_steps}")
    
    async with trainer.training_session():
        for step in range(total_steps):
            batch = {
                'size': 32,
                'data': f"batch_{step}",
                'step': step
            }
            
            try:
                result = await trainer.train_step(batch)
                
                if step % 10 == 0:
                    print(f"  Step {trainer.step:3d}: loss={result['loss']:.3f}, acc={result['accuracy']:.3f}")
                
            except Exception as e:
                print(f"  ❌ Step {step} failed: {e}")
                # Training should recover automatically
                continue
    
    # Check how many checkpoints were created
    if trainer.resilience_manager:
        checkpoint_count = len(trainer.resilience_manager.checkpoints)
        print(f"✅ Created {checkpoint_count} checkpoints")
        print(f"📊 Expected: ~{total_steps // checkpoint_interval} checkpoints")
        
        if checkpoint_count > 0:
            latest_checkpoint = max(
                trainer.resilience_manager.checkpoints.values(),
                key=lambda x: x.timestamp
            )
            print(f"📝 Latest checkpoint: {latest_checkpoint.checkpoint_id}")
            print(f"   Step: {latest_checkpoint.step}")
            print(f"   Size: {latest_checkpoint.file_size_bytes:,} bytes")
    
    print()


async def test_failure_recovery(trainer, model, optimizer):
    """Test failure recovery mechanisms."""
    print("🛡️ TESTING FAILURE RECOVERY")
    print("-" * 30)
    
    # Increase failure rate for testing
    original_failure_rate = model.failure_rate
    model.failure_rate = 0.15  # 15% failure rate
    
    print(f"⚠️  Simulated failure rate: {model.failure_rate:.1%}")
    print("🔄 Running training with failures...")
    
    failure_count = 0
    recovery_count = 0
    successful_steps = 0
    
    async with trainer.training_session():
        for step in range(50):
            batch = {'size': 32, 'data': f'batch_{step}'}
            
            try:
                result = await trainer.train_step(batch)
                successful_steps += 1
                
                if step % 10 == 0:
                    print(f"  ✅ Step {trainer.step}: loss={result['loss']:.3f}")
                
            except Exception as e:
                failure_count += 1
                print(f"  ❌ Step {step} failed: {type(e).__name__}")
                
                # Check if resilience manager handled it
                if trainer.resilience_manager:
                    status = trainer.resilience_manager.get_resilience_status()
                    total_failures = status['total_failures']
                    if total_failures > 0:
                        recovery_count += 1
                        print(f"    🔧 Recovery attempted (total failures: {total_failures})")
    
    # Restore original failure rate
    model.failure_rate = original_failure_rate
    
    print(f"\n📊 FAILURE RECOVERY RESULTS:")
    print(f"   Successful steps: {successful_steps}")
    print(f"   Failed steps: {failure_count}")
    print(f"   Recovery attempts: {recovery_count}")
    print(f"   Success rate: {successful_steps/(successful_steps + failure_count):.1%}")
    
    if trainer.resilience_manager:
        status = trainer.resilience_manager.get_resilience_status()
        print(f"   Health status: {status['health_status']['overall']}")
        print(f"   Circuit breakers: {len(status['circuit_breakers'])} active")
    
    print()


async def test_checkpoint_recovery(trainer, model, optimizer):
    """Test checkpoint-based recovery."""
    print("💾 TESTING CHECKPOINT RECOVERY")
    print("-" * 35)
    
    if not trainer.resilience_manager:
        print("❌ Resilience manager not available, skipping test")
        return
    
    # Train for a bit and create checkpoints
    print("🏃 Training to create checkpoints...")
    async with trainer.training_session():
        for step in range(30):
            batch = {'size': 16, 'data': f'recovery_batch_{step}'}
            try:
                await trainer.train_step(batch)
            except Exception:
                pass  # Ignore failures for this test
    
    # Get the checkpoints
    checkpoints = list(trainer.resilience_manager.checkpoints.keys())
    if not checkpoints:
        print("❌ No checkpoints available for recovery test")
        return
    
    # Pick a recent checkpoint
    latest_checkpoint_id = checkpoints[-1]
    latest_checkpoint = trainer.resilience_manager.checkpoints[latest_checkpoint_id]
    
    print(f"📦 Testing recovery from checkpoint: {latest_checkpoint_id}")
    print(f"   Created at step: {latest_checkpoint.step}")
    print(f"   Timestamp: {latest_checkpoint.timestamp.strftime('%H:%M:%S')}")
    
    # Save current state
    current_step = trainer.step
    current_model_step = model.training_step
    
    try:
        # Restore checkpoint
        checkpoint_data = await trainer.resilience_manager.restore_checkpoint(latest_checkpoint_id)
        
        print("✅ Checkpoint restored successfully")
        print(f"   Checkpoint contains: {list(checkpoint_data.keys())}")
        
        # Verify data integrity
        if 'model_state' in checkpoint_data and checkpoint_data['model_state']:
            print("✅ Model state recovered")
        
        if 'optimizer_state' in checkpoint_data and checkpoint_data['optimizer_state']:
            print("✅ Optimizer state recovered")
        
        if 'metrics' in checkpoint_data:
            metrics = checkpoint_data['metrics']
            print(f"✅ Metrics recovered: {len(metrics)} fields")
            print(f"   Recovered step: {checkpoint_data.get('step', 'unknown')}")
            print(f"   Recovered epoch: {checkpoint_data.get('epoch', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Checkpoint recovery failed: {e}")
    
    print()


async def test_health_monitoring(trainer):
    """Test health monitoring system."""
    print("🏥 TESTING HEALTH MONITORING")
    print("-" * 30)
    
    if not trainer.resilience_manager:
        print("❌ Resilience manager not available, skipping test")
        return
    
    print("📊 Current health status:")
    status = trainer.resilience_manager.get_resilience_status()
    health = status['health_status']
    
    for component, state in health.items():
        if component == 'last_check':
            continue
        print(f"   {component:15}: {state}")
    
    print(f"   last_check: {health.get('last_check', 'unknown')}")
    print()
    
    print("🔍 Running health check...")
    await trainer.resilience_manager._perform_health_check()
    
    # Get updated status
    updated_status = trainer.resilience_manager.get_resilience_status()
    updated_health = updated_status['health_status']
    
    print("📊 Updated health status:")
    for component, state in updated_health.items():
        if component == 'last_check':
            continue
        print(f"   {component:15}: {state}")
    
    # Check circuit breakers
    circuit_breakers = updated_status['circuit_breakers']
    print(f"\n⚡ Circuit breakers: {len(circuit_breakers)}")
    for service, info in circuit_breakers.items():
        print(f"   {service:15}: {info['state']} (failures: {info['failure_count']})")
    
    print()


async def robustness_demo():
    """Comprehensive robustness demonstration."""
    print("🛡️ GENERATION 2 ROBUSTNESS DEMO")
    print("=" * 50)
    
    # Create robust model and optimizer with failure simulation
    model = RobustDemoModel("robust-transformer", failure_rate=0.02)  # 2% failure rate
    optimizer = RobustDemoOptimizer(lr=0.001)
    
    # Configure comprehensive resilience
    config = TrainingConfig(
        carbon_threshold=100.0,
        pause_threshold=150.0,
        resume_threshold=70.0,
        check_interval=30,  # Quick checks for demo
        
        # Generation 2: Resilience settings
        enable_resilience=True,
        max_failures=10,           # Allow more failures for demo
        checkpoint_interval=25,    # Checkpoint every 25 steps
        max_recovery_time=timedelta(minutes=5),
        health_check_interval=30,  # 30 second health checks
        retry_attempts=3,
        circuit_breaker_enabled=True,
    )
    
    print(f"🤖 Model: {model.name}")
    print(f"⚡ Parameters: {model.parameters:,}")
    print(f"🎯 Failure simulation: {model.failure_rate:.1%} per step")
    print(f"🛡️ Max failures: {config.max_failures}")
    print(f"💾 Checkpoint interval: {config.checkpoint_interval} steps")
    print()
    
    # Initialize robust trainer
    trainer = CarbonAwareTrainer(
        model=model,
        optimizer=optimizer,
        carbon_model='cached',  # Use cached for demo
        region='US-CA',
        config=config
    )
    
    trainer.adaptive_training = True  # Keep Generation 1 features
    
    try:
        print("🚀 Initializing robust trainer...")
        await trainer.initialize()
        
        if trainer.resilience_manager:
            print("✅ Resilience manager active")
            print("✅ Health monitoring enabled")
            print("✅ Automatic checkpointing enabled")
            print("✅ Circuit breaker protection enabled")
        else:
            print("⚠️ Resilience manager not available (missing dependencies)")
        
        print()
        
        # Run comprehensive robustness tests
        await test_health_monitoring(trainer)
        await test_automatic_checkpointing(trainer, model, optimizer)
        await test_failure_recovery(trainer, model, optimizer) 
        await test_checkpoint_recovery(trainer, model, optimizer)
        
        # Final resilience status
        print("📊 FINAL RESILIENCE STATUS")
        print("-" * 30)
        
        if trainer.resilience_manager:
            final_status = trainer.resilience_manager.get_resilience_status()
            
            print(f"🏥 Overall health: {final_status['health_status']['overall']}")
            print(f"❌ Total failures: {final_status['total_failures']}")
            print(f"💾 Checkpoints created: {final_status['checkpoint_count']}")
            print(f"🔧 Recovery events: {final_status['recovery_history_count']}")
            
            failure_counts = final_status['failure_counts']
            if failure_counts:
                print("📊 Failure breakdown:")
                for failure_type, count in failure_counts.items():
                    print(f"   {failure_type:15}: {count}")
        
        # Final training metrics
        print("\n📈 FINAL TRAINING METRICS")
        print("-" * 30)
        
        final_metrics = trainer.get_carbon_metrics()
        print(f"🔢 Total steps: {final_metrics['step']}")
        print(f"⏱️ Runtime: {final_metrics['runtime_hours']:.2f} hours") 
        print(f"⚡ Energy: {final_metrics['total_energy_kwh']:.4f} kWh")
        print(f"🌱 Carbon: {final_metrics['total_carbon_kg']:.4f} kg CO2")
        print(f"📊 Efficiency: {final_metrics['efficiency_score']:.1f}/100")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await trainer.cleanup()
    
    print("\n✅ Generation 2 robustness demo completed!")
    print("🎉 Robustness features: Automatic checkpointing, failure recovery, health monitoring")


async def quick_resilience_test():
    """Quick test of resilience features."""
    print("\n🔍 QUICK RESILIENCE TEST")
    print("=" * 30)
    
    # Create simple test setup
    model = RobustDemoModel(failure_rate=0.0)  # No failures for basic test
    config = TrainingConfig(
        enable_resilience=True,
        checkpoint_interval=10,
        max_failures=3
    )
    
    trainer = CarbonAwareTrainer(
        model=model,
        carbon_model='cached',
        region='US-CA',
        config=config
    )
    
    print("🚀 Testing basic resilience setup...")
    await trainer.initialize()
    
    if trainer.resilience_manager:
        print("✅ Resilience manager initialized")
        
        # Test health check
        health = trainer.resilience_manager.get_resilience_status()
        print(f"✅ Health monitoring: {health['health_status']['overall']}")
        
        # Test checkpoint creation
        await trainer._create_resilient_checkpoint()
        checkpoint_count = len(trainer.resilience_manager.checkpoints)
        print(f"✅ Checkpoint system: {checkpoint_count} checkpoint(s) created")
        
    else:
        print("⚠️ Resilience manager not available")
    
    await trainer.cleanup()
    print("✅ Resilience test completed!")


async def main():
    """Main demo function."""
    print("🌱 CARBON-AWARE TRAINER - GENERATION 2")
    print("🛡️ Robustness & Reliability Demo")
    print("=" * 60)
    print()
    
    # Run quick resilience test first
    await quick_resilience_test()
    
    # Run comprehensive demo
    await robustness_demo()
    
    print("\n" + "=" * 60)
    print("🎯 GENERATION 2 COMPLETE")
    print("✨ Robustness features added:")
    print("   • Automatic checkpointing and recovery")
    print("   • Comprehensive error handling and retry logic")
    print("   • Circuit breaker patterns for external services")
    print("   • Health monitoring and failure detection")  
    print("   • Production-ready resilience management")
    print("   • Intelligent recovery planning and execution")
    print()
    print("➡️ Ready for Generation 3: Optimization & Scaling")


if __name__ == "__main__":
    asyncio.run(main())