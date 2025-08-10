#!/usr/bin/env python3
"""Production robustness demonstration with comprehensive error handling."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, timedelta
import json
import logging
import random

from carbon_aware_trainer import CarbonAwareTrainer
from carbon_aware_trainer.core.types import TrainingConfig
from carbon_aware_trainer.core.robustness import RobustnessManager, HealthStatus


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RobustTrainingModel:
    """Production-ready model with failure simulation."""
    
    def __init__(self, failure_rate=0.1):
        self.step_count = 0
        self.failure_rate = failure_rate
        self.total_energy = 0
    
    def train_step(self, batch, **kwargs):
        """Training step with potential failures."""
        self.step_count += 1
        
        # Simulate occasional failures
        if random.random() < self.failure_rate:
            raise RuntimeError(f"Simulated training failure at step {self.step_count}")
        
        # Simulate varying computational load
        complexity = random.uniform(0.5, 2.0)
        energy_consumption = 0.001 * complexity  # kWh
        self.total_energy += energy_consumption
        
        return {
            'loss': max(0.01, 2.0 - (self.step_count * 0.05)),
            'accuracy': min(0.98, self.step_count * 0.02),
            'energy_kwh': energy_consumption,
            'complexity': complexity
        }


async def demonstrate_robustness():
    """Demonstrate production robustness features."""
    print("🛡️  Production Robustness Demonstration")
    print("=" * 50)
    
    # Initialize robustness manager
    robustness = RobustnessManager(
        health_check_interval=5,    # Check every 5 seconds
        max_retry_attempts=3,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=30
    )
    
    # Add comprehensive monitoring callbacks
    def health_callback(event_type, data):
        if event_type == 'health_update':
            overall = data['overall_status']
            print(f"🏥 Health: {overall.value.upper()}")
            
            for component, check in data['component_checks'].items():
                status_emoji = {
                    'healthy': '✅',
                    'warning': '⚠️',
                    'critical': '🚨',
                    'degraded': '🔧'
                }.get(check.status.value, '❓')
                
                print(f"   {status_emoji} {component}: {check.message}")
    
    def alert_callback(event_type, data):
        if event_type == 'health_alert':
            print(f"🚨 ALERT - {data['component']}: {data['message']}")
            for suggestion in data['suggestions']:
                print(f"   💡 {suggestion}")
    
    robustness.add_health_callback(health_callback)
    robustness.add_alert_callback(alert_callback)
    
    # Start health monitoring
    await robustness.start_health_monitoring()
    
    # Create model with some failures
    model = RobustTrainingModel(failure_rate=0.15)  # 15% failure rate
    
    # Robust training configuration
    config = TrainingConfig(
        carbon_threshold=100.0,
        pause_threshold=200.0,
        resume_threshold=75.0,
        check_interval=10
    )
    
    # Initialize trainer
    trainer = CarbonAwareTrainer(
        model=model,
        carbon_model='cached',
        region='US-CA',
        config=config,
        api_key='sample_data/sample_carbon_data.json'
    )
    
    # Enhanced callbacks with robustness tracking
    def training_callback(state, metrics):
        # Record operation success/failure for robustness tracking
        robustness.record_operation(success=True)
        
        print(f"🔄 Training: {state.value} | "
              f"Steps: {trainer.step} | "
              f"Carbon: {metrics.total_carbon_kg:.3f} kg CO2")
    
    trainer.add_state_callback(training_callback)
    
    print("🚀 Starting Robust Training Session...")
    
    try:
        async with trainer:
            await trainer.start_training()
            
            total_steps = 30
            successful_steps = 0
            failed_steps = 0
            retry_attempts = 0
            
            for step in range(total_steps):
                batch = {
                    'data': f'robust_batch_{step}',
                    'size': random.randint(16, 64),
                    'timestamp': datetime.now()
                }
                
                # Implement retry logic with exponential backoff
                max_retries = 3
                base_delay = 1.0
                
                for attempt in range(max_retries + 1):
                    try:
                        result = await trainer.train_step(batch)
                        
                        # Record successful operation
                        robustness.record_operation(success=True)
                        successful_steps += 1
                        
                        if step % 5 == 0:
                            print(f"   ✅ Step {step}: "
                                  f"Loss {result['loss']:.3f}, "
                                  f"Accuracy {result['accuracy']:.3f}, "
                                  f"Energy {result['energy_kwh']:.4f} kWh, "
                                  f"Complexity {result['complexity']:.1f}x")
                        
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        # Record failed operation
                        robustness.record_operation(success=False)
                        robustness.record_failure('training_step')
                        
                        retry_attempts += 1
                        
                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            print(f"   ⚠️  Step {step} failed (attempt {attempt + 1}): {e}")
                            print(f"   🔄 Retrying in {delay:.1f}s...")
                            await asyncio.sleep(delay)
                        else:
                            print(f"   ❌ Step {step} failed permanently after {max_retries + 1} attempts")
                            failed_steps += 1
                            break
                
                # Check if circuit breaker is open
                if robustness.is_circuit_breaker_open('training_step'):
                    print("   🔌 Circuit breaker open - pausing training")
                    await asyncio.sleep(5)
                
                # Small delay between steps
                await asyncio.sleep(0.2)
                
                # Periodic health summary
                if step % 10 == 0 and step > 0:
                    health_summary = robustness.get_health_summary()
                    print(f"\n📊 Health Summary (Step {step}):")
                    print(f"   Overall: {health_summary['overall_status'].upper()}")
                    print(f"   Uptime: {health_summary['uptime_hours']:.2f}h")
                    print(f"   Success Rate: {health_summary['success_rate']:.1%}")
                    print(f"   Operations: {health_summary['total_operations']}")
                    if health_summary['circuit_breakers']['open_count'] > 0:
                        print(f"   🔌 Open Breakers: {health_summary['circuit_breakers']['open_components']}")
                    print()
            
            # Final comprehensive metrics
            final_metrics = trainer.get_carbon_metrics()
            health_summary = robustness.get_health_summary()
            
            print("\n" + "=" * 60)
            print("📊 PRODUCTION ROBUSTNESS REPORT")
            print("=" * 60)
            
            # Training results
            print(f"🎯 Training Results:")
            print(f"   Total Steps Attempted: {total_steps}")
            print(f"   Successful Steps: {successful_steps}")
            print(f"   Failed Steps: {failed_steps}")
            print(f"   Retry Attempts: {retry_attempts}")
            print(f"   Success Rate: {(successful_steps/total_steps):.1%}")
            
            # Carbon metrics
            print(f"\n🌱 Carbon Metrics:")
            print(f"   Total Energy: {final_metrics['total_energy_kwh']:.4f} kWh")
            print(f"   Total Carbon: {final_metrics['total_carbon_kg']:.3f} kg CO2")
            print(f"   Runtime: {final_metrics['runtime_hours']:.3f} hours")
            
            # Robustness metrics
            print(f"\n🛡️  Robustness Metrics:")
            print(f"   Overall Health: {health_summary['overall_status'].upper()}")
            print(f"   System Uptime: {health_summary['uptime_hours']:.2f} hours")
            print(f"   Operation Success Rate: {health_summary['success_rate']:.1%}")
            print(f"   Total Operations: {health_summary['total_operations']}")
            print(f"   Circuit Breakers Open: {health_summary['circuit_breakers']['open_count']}")
            print(f"   Recent Auto-Recoveries: {health_summary['recent_recoveries']}")
            
            # Component health details
            print(f"\n🔧 Component Health:")
            for component, health in health_summary['component_health'].items():
                status_emoji = {
                    'healthy': '✅',
                    'warning': '⚠️', 
                    'critical': '🚨',
                    'degraded': '🔧'
                }.get(health['status'], '❓')
                
                print(f"   {status_emoji} {component}: {health['message']}")
                
                # Show key metrics
                if health['metrics']:
                    for metric, value in health['metrics'].items():
                        if isinstance(value, float):
                            print(f"      📈 {metric}: {value:.2f}")
                        else:
                            print(f"      📈 {metric}: {value}")
    
    except Exception as e:
        logger.error(f"Training session failed: {e}")
        robustness.record_operation(success=False)
        print(f"❌ Training session failed: {e}")
    
    finally:
        await robustness.stop_health_monitoring()
    
    print("\n✅ Production robustness demonstration completed!")
    print("🛡️  System demonstrated resilience against failures and monitoring capabilities")


if __name__ == "__main__":
    asyncio.run(demonstrate_robustness())