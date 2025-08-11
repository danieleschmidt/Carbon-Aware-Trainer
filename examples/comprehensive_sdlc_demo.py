#!/usr/bin/env python3
"""
Comprehensive SDLC Demo - All Three Generations

This demo showcases the complete Carbon-Aware-Trainer system:
- Generation 1: Advanced features (multi-region, real-time optimization, federated learning)
- Generation 2: Robustness (circuit breakers, validation, health monitoring)
- Generation 3: Scaling (performance optimization, intelligent caching, auto-scaling)
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta

from carbon_aware_trainer import CarbonMonitor, CarbonAwareTrainer
from carbon_aware_trainer.core.types import CarbonDataSource

# Import Generation 1 features
from carbon_aware_trainer.core.multi_region import MultiRegionOrchestrator, RegionConfig
from carbon_aware_trainer.core.real_time_optimizer import RealTimeOptimizer, OptimizationMode

# Import Generation 2 features  
from carbon_aware_trainer.core.circuit_breaker import circuit_breaker_manager, CircuitBreakerConfig
from carbon_aware_trainer.core.comprehensive_validation import validator
from carbon_aware_trainer.core.health_monitoring import health_monitor

# Import Generation 3 features
from carbon_aware_trainer.core.performance_optimizer import PerformanceOptimizer, OptimizationStrategy
from carbon_aware_trainer.core.intelligent_scaling import IntelligentAutoScaler, ScalingMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def comprehensive_training_simulation():
    """Simulate a comprehensive carbon-aware training workflow."""
    print("\n" + "🌟 " + "="*58 + " 🌟")
    print("   COMPREHENSIVE CARBON-AWARE TRAINING SIMULATION")
    print("🌟 " + "="*58 + " 🌟")
    
    print("\n📋 Initializing all system components...")
    
    # Initialize core monitor
    monitor = CarbonMonitor(
        regions=["US-CA", "US-WA"],
        data_source=CarbonDataSource.CACHED,
        api_key="sample_data/sample_carbon_data.json"
    )
    
    # Initialize Generation 1 components
    regions = {
        "US-CA": RegionConfig(
            region_id="US-CA",
            gpus=8,
            cost_per_hour=10.0,
            bandwidth_gbps=10,
            carbon_threshold=120
        ),
        "US-WA": RegionConfig(
            region_id="US-WA",
            gpus=4,
            cost_per_hour=8.0,
            bandwidth_gbps=8,
            carbon_threshold=100
        )
    }
    
    orchestrator = MultiRegionOrchestrator(regions=regions, monitor=monitor)
    optimizer = RealTimeOptimizer(monitor=monitor, mode=OptimizationMode.BALANCED)
    
    # Initialize Generation 2 components
    await health_monitor.start_monitoring()
    
    # Initialize Generation 3 components
    performance_optimizer = PerformanceOptimizer(strategy=OptimizationStrategy.BALANCED)
    scaler = IntelligentAutoScaler(enable_carbon_aware=True, enable_predictive=True)
    
    print("✅ All system components initialized")
    
    # Phase 1: Pre-training planning and validation
    print("\n🎯 Phase 1: Pre-training Planning & Validation")
    
    # Validate system configuration
    validation_data = {
        "memory_gb": 32,
        "gpu_count": 8,
        "training_config": {"carbon_threshold": 120, "batch_size": 64},
        "user_inputs": "normal training parameters"
    }
    
    validation_result = validator.run_comprehensive_validation(**validation_data)
    validation_summary = validator.get_validation_summary(validation_result)
    
    print(f"   📊 System Validation: {'✅ PASSED' if validation_summary['is_valid'] else '❌ FAILED'}")
    if validation_summary['total_issues'] > 0:
        print(f"      Issues found: {validation_summary['total_issues']}")
        for issue in validation_summary.get('error_issues', [])[:2]:
            print(f"         🚨 {issue}")
    
    # Plan optimal placement
    placement_plan = await orchestrator.optimize_placement(
        model_size_gb=25,
        dataset_size_gb=200,
        training_hours=48,
        carbon_budget_kg=500
    )
    
    print(f"   🌍 Multi-region placement optimized:")
    print(f"      Primary: {placement_plan.primary_region}")
    print(f"      Expected carbon: {placement_plan.expected_carbon_kg:.1f} kg CO2")
    print(f"      Carbon savings: {placement_plan.carbon_savings_pct:.1f}%")
    
    # Phase 2: Training execution with real-time optimization
    print("\n🚀 Phase 2: Real-time Training Execution")
    
    # Start real-time optimization
    async def training_callback(action):
        print(f"      ⚡ Optimization action: {action.action_type} ({action.confidence:.2f} confidence)")
    
    await optimizer.start_optimization(region=placement_plan.primary_region, training_callback=training_callback)
    
    # Simulate training workload with performance optimization
    training_data = list(range(500))  # Simulate 500 training samples
    
    print("   🧠 Processing training batch with performance optimization...")
    
    def dummy_training_function(batch):
        """Simulate training computation."""
        time.sleep(0.01)  # Simulate computation
        return [x * 2 + 1 for x in batch[:10]]  # Process first 10 items
    
    start_time = time.time()
    processed_data, perf_report = await performance_optimizer.optimize_workload(
        dummy_training_function,
        training_data,
        target_latency_ms=100,
        target_throughput=100
    )
    execution_time = time.time() - start_time
    
    print(f"      📈 Batch processed: {len(processed_data)} items in {execution_time:.2f}s")
    print(f"      📊 Throughput: {perf_report['throughput_items_sec']:.1f} items/sec")
    print(f"      🎯 Cache hit rate: {perf_report['cache_hit_rate']:.1%}")
    
    # Phase 3: Adaptive scaling simulation
    print("\n📈 Phase 3: Adaptive Scaling Simulation")
    
    print("   🔄 Simulating varying workload demands...")
    
    # Simulate 10 minutes of varying load
    for minute in range(10):
        # Vary load based on time
        if minute < 3:
            cpu_util = 45 + (minute * 10)  # Gradual increase
            carbon_intensity = 120
        elif minute < 6:
            cpu_util = 80 + (minute * 2)   # High load
            carbon_intensity = 200 + (minute * 30)
        else:
            cpu_util = 60 - ((minute - 6) * 8)  # Decrease
            carbon_intensity = 80
        
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=cpu_util,
            memory_utilization=cpu_util * 0.8,
            carbon_intensity=carbon_intensity,
            queue_depth=max(0, int((cpu_util - 50) / 5)),
            response_time_p95=200 + (cpu_util - 50) * 8,
            active_requests=int(cpu_util * 1.5)
        )
        
        await scaler.update_metrics(metrics)
        
        scaling_status = scaler.get_scaling_status()
        print(f"   Min {minute+1}: CPU {cpu_util:.0f}%, Carbon {carbon_intensity} gCO2/kWh, "
              f"Instances: {scaling_status['current_instances']}")
        
        await asyncio.sleep(0.1)  # Brief pause
    
    # Phase 4: Health monitoring and circuit breaker demonstration
    print("\n💚 Phase 4: System Health & Resilience")
    
    # Get health status
    health_status = await health_monitor.get_health_status()
    print(f"   🏥 System Health: {health_status['overall_status'].upper()}")
    print(f"      Active alerts: {health_status['active_alerts']}")
    
    if 'system_metrics' in health_status:
        metrics = health_status['system_metrics']
        if 'cpu' in metrics and 'error' not in metrics['cpu']:
            print(f"      CPU usage: {metrics['cpu']['cpu_percent']:.1f}%")
        if 'memory' in metrics and 'error' not in metrics['memory']:
            print(f"      Memory usage: {metrics['memory']['memory_percent']:.1f}%")
    
    # Test circuit breaker resilience
    print("   🔄 Testing system resilience with circuit breakers...")
    
    # Create a test circuit breaker
    test_breaker = circuit_breaker_manager.get_breaker(
        "training_service",
        CircuitBreakerConfig(failure_threshold=2, recovery_timeout=3.0)
    )
    
    # Simulate some failures and recovery
    for attempt in range(5):
        try:
            if attempt < 3:
                # Simulate failures
                await test_breaker.call(lambda: (_ for _ in ()).throw(Exception("Service down")))
            else:
                # Simulate recovery
                result = await test_breaker.call(lambda: "Service healthy")
                print(f"      ✅ Circuit breaker test {attempt+1}: {result}")
        except Exception as e:
            print(f"      ⚠️ Circuit breaker test {attempt+1}: {type(e).__name__}")
        
        await asyncio.sleep(0.5)
    
    # Phase 5: Final system summary
    print("\n📊 Phase 5: System Performance Summary")
    
    # Performance optimizer summary
    perf_summary = performance_optimizer.get_optimization_summary()
    print(f"   ⚡ Performance Optimizer:")
    print(f"      Strategy: {perf_summary['strategy']}")
    print(f"      Optimization runs: {perf_summary['optimization_runs']}")
    
    # Scaling summary
    scaling_summary = scaler.get_scaling_status()
    print(f"   📈 Auto-scaler:")
    print(f"      Final instances: {scaling_summary['current_instances']}")
    print(f"      Total scaling actions: {scaling_summary['total_scaling_actions']}")
    print(f"      Carbon efficiency: {scaling_summary['carbon_efficiency_score']:.2f}")
    
    # Circuit breaker summary
    circuit_summary = circuit_breaker_manager.get_health_summary()
    print(f"   🔄 Circuit Breakers:")
    print(f"      Status: {circuit_summary['status']}")
    print(f"      Total breakers: {circuit_summary['total_breakers']}")
    
    # Health monitoring summary
    recent_alerts = health_monitor.get_alerts(since_hours=1)
    print(f"   💚 Health Monitor:")
    print(f"      Recent alerts: {len(recent_alerts)}")
    print(f"      Overall status: {health_status['overall_status']}")
    
    # Cleanup
    await optimizer.stop_optimization()
    await health_monitor.stop_monitoring()
    performance_optimizer.cleanup()
    
    print("\n🎉 COMPREHENSIVE SIMULATION COMPLETED SUCCESSFULLY! 🎉")
    print("\nThe Carbon-Aware-Trainer system demonstrates:")
    print("  ✅ Generation 1: Advanced multi-region orchestration & real-time optimization")
    print("  ✅ Generation 2: Robust error handling, validation & health monitoring")
    print("  ✅ Generation 3: Intelligent performance optimization & auto-scaling")
    print("\n🌍 Ready for production-scale carbon-aware ML training! 🌍")


async def quick_functionality_test():
    """Quick test of core functionality."""
    print("\n🔬 Quick Functionality Test")
    print("="*40)
    
    try:
        # Test basic carbon monitoring
        monitor = CarbonMonitor(
            regions=["US-CA"],
            data_source=CarbonDataSource.CACHED,
            api_key="sample_data/sample_carbon_data.json"
        )
        
        current = await monitor.get_current_intensity("US-CA")
        print(f"✅ Carbon monitoring: {current.carbon_intensity} gCO2/kWh")
        
        # Test trainer
        trainer = CarbonAwareTrainer(
            region="US-CA",
            carbon_model="cached",
            target_carbon_intensity=100.0
        )
        
        metrics = trainer.get_carbon_metrics()
        print(f"✅ Training metrics: {len(metrics)} metrics available")
        
        # Test validation
        result = validator.validate_api_input({"test_key": "safe_value"})
        print(f"✅ Validation system: {result.is_valid}")
        
        print("✅ All core systems operational!")
        
    except Exception as e:
        print(f"❌ Error in functionality test: {e}")
        return False
        
    return True


async def main():
    """Run the comprehensive SDLC demonstration."""
    print("🚀 Carbon-Aware-Trainer - Autonomous SDLC Implementation")
    print("=" * 60)
    print("Demonstrating all three generations of the system:")
    print("  Generation 1: Make it Work (Advanced Features)")
    print("  Generation 2: Make it Robust (Reliability & Monitoring)")
    print("  Generation 3: Make it Scale (Optimization & Auto-scaling)")
    
    # Quick functionality test first
    functionality_ok = await quick_functionality_test()
    
    if functionality_ok:
        # Run comprehensive simulation
        await comprehensive_training_simulation()
    else:
        print("\n❌ Basic functionality test failed - skipping comprehensive demo")
        return 1
        
    return 0


if __name__ == "__main__":
    # Run the comprehensive demo
    result = asyncio.run(main())
    exit(result)