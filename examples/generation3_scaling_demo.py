#!/usr/bin/env python3
"""
Generation 3 Scaling & Optimization Demo

Demonstrates the advanced scaling and optimization features:
- Performance optimization with adaptive resource management
- Intelligent caching system
- Predictive auto-scaling
- Carbon-aware scaling decisions
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta

from carbon_aware_trainer.core.performance_optimizer import (
    PerformanceOptimizer, OptimizationStrategy, performance_optimizer
)
from carbon_aware_trainer.core.intelligent_scaling import (
    IntelligentAutoScaler, ScalingMetrics, ScalingRule, ScalingTrigger,
    intelligent_scaler
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_workload(items):
    """Simulate a CPU-intensive workload."""
    # Simulate processing delay
    time.sleep(random.uniform(0.1, 0.3))
    
    # Process items (simple computation)
    results = []
    for item in items:
        # Simulate some computation
        result = sum(range(item % 100)) if isinstance(item, int) else len(str(item))
        results.append(result)
    
    return results


def simulate_complex_workload(data):
    """Simulate a more complex workload for performance testing."""
    # Simulate varying computational complexity
    complexity = random.uniform(0.05, 0.5)
    time.sleep(complexity)
    
    # Return processed data
    if isinstance(data, list):
        return [x * 2 + random.randint(1, 10) for x in data[:10]]  # Limit processing
    else:
        return data * 2 + random.randint(1, 10)


async def demo_performance_optimization():
    """Demo advanced performance optimization."""
    print("\n" + "="*60)
    print("⚡ Advanced Performance Optimization Demo")
    print("="*60)
    
    # Test different optimization strategies
    strategies = [
        OptimizationStrategy.THROUGHPUT,
        OptimizationStrategy.LATENCY,
        OptimizationStrategy.EFFICIENCY,
        OptimizationStrategy.BALANCED
    ]
    
    # Generate test workload
    test_data = list(range(1000))  # 1000 items to process
    
    results = {}
    
    for strategy in strategies:
        print(f"\n🧪 Testing {strategy.value} optimization strategy...")
        
        # Create optimizer for this strategy
        optimizer = PerformanceOptimizer(strategy=strategy)
        
        # Run optimization
        start_time = time.time()
        processed_data, performance_report = await optimizer.optimize_workload(
            simulate_complex_workload,
            test_data,
            target_latency_ms=500,
            target_throughput=100
        )
        
        execution_time = time.time() - start_time
        
        # Store results
        results[strategy.value] = {
            "execution_time": execution_time,
            "performance_report": performance_report,
            "items_processed": len(processed_data)
        }
        
        print(f"✅ Completed in {execution_time:.2f}s")
        print(f"   Items processed: {len(processed_data)}")
        print(f"   Throughput: {performance_report['throughput_items_sec']:.1f} items/sec")
        print(f"   Cache hit rate: {performance_report['cache_hit_rate']:.1%}")
        print(f"   Batch size used: {performance_report['execution_plan']['batch_size']}")
        
        if performance_report['recommendations']:
            print(f"   📋 Recommendations:")
            for rec in performance_report['recommendations'][:2]:  # Show top 2
                print(f"     • {rec['message']}")
                
        # Clean up
        optimizer.cleanup()
        
    # Compare strategies
    print(f"\n📊 Strategy Comparison:")
    print(f"{'Strategy':<12} {'Time (s)':<10} {'Throughput':<12} {'Cache Hit':<10}")
    print("-" * 50)
    
    for strategy_name, result in results.items():
        print(f"{strategy_name:<12} {result['execution_time']:<10.2f} "
              f"{result['performance_report']['throughput_items_sec']:<12.1f} "
              f"{result['performance_report']['cache_hit_rate']:<10.1%}")


async def demo_intelligent_caching():
    """Demo intelligent caching system."""
    print("\n" + "="*60)
    print("🧠 Intelligent Caching System Demo")
    print("="*60)
    
    optimizer = PerformanceOptimizer()
    cache = optimizer.cache_manager
    
    print("📝 Testing cache performance with different access patterns...")
    
    # Test 1: Sequential access
    print("\n🔄 Test 1: Sequential Access Pattern")
    for i in range(50):
        cache.set(f"key_{i}", f"value_{i}")
        
    # Access some keys multiple times
    for i in range(20):
        for key in [f"key_{i}" for i in range(0, 50, 5)]:  # Every 5th key
            value = cache.get(key)
            
    stats = cache.get_stats()
    print(f"   Cache entries: {stats['entries']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"   Memory usage: {stats['memory_usage_mb']:.2f} MB")
    
    # Test 2: Hot/Cold access pattern
    print("\n🔥 Test 2: Hot/Cold Access Pattern")
    cache.clear()
    
    # Create hot keys (frequently accessed)
    hot_keys = [f"hot_{i}" for i in range(10)]
    cold_keys = [f"cold_{i}" for i in range(100)]
    
    # Store all keys
    for key in hot_keys + cold_keys:
        cache.set(key, f"data_for_{key}")
        
    # Access hot keys frequently, cold keys rarely
    for _ in range(100):
        # 80% chance to access hot key, 20% for cold key
        if random.random() < 0.8:
            key = random.choice(hot_keys)
        else:
            key = random.choice(cold_keys)
        cache.get(key)
        
    final_stats = cache.get_stats()
    print(f"   Final hit rate: {final_stats['hit_rate']:.1%}")
    print(f"   Total requests: {final_stats['total_requests']}")
    print(f"   Evictions: {final_stats['evictions']}")
    
    # Test 3: Memory pressure
    print("\n💾 Test 3: Memory Pressure Handling")
    cache.clear()
    
    # Fill cache beyond limit to trigger evictions
    for i in range(200):
        large_data = "x" * 1000  # 1KB per entry
        cache.set(f"large_{i}", large_data)
        
    pressure_stats = cache.get_stats()
    print(f"   Entries after memory pressure: {pressure_stats['entries']}")
    print(f"   Memory utilization: {pressure_stats['memory_utilization']:.1%}")
    print(f"   Total evictions: {pressure_stats['evictions']}")
    
    optimizer.cleanup()


async def demo_intelligent_scaling():
    """Demo intelligent auto-scaling system."""
    print("\n" + "="*60)
    print("🚀 Intelligent Auto-Scaling Demo")
    print("="*60)
    
    # Create scaler with carbon awareness
    scaler = IntelligentAutoScaler(
        enable_predictive=True,
        enable_carbon_aware=True,
        carbon_weight=0.4
    )
    
    print("📊 Simulating workload with varying resource demands...")
    
    # Simulate 30 minutes of metrics (1 metric per minute)
    for minute in range(30):
        print(f"\n⏰ Minute {minute + 1}/30")
        
        # Simulate different load patterns
        if minute < 10:
            # Normal load
            cpu_util = random.uniform(40, 60)
            memory_util = random.uniform(30, 50)
            carbon_intensity = random.uniform(100, 150)
        elif minute < 20:
            # High load period
            cpu_util = random.uniform(70, 90)
            memory_util = random.uniform(60, 80)
            carbon_intensity = random.uniform(200, 400)
        else:
            # Low load with clean energy
            cpu_util = random.uniform(20, 40)
            memory_util = random.uniform(20, 40)
            carbon_intensity = random.uniform(30, 80)
            
        # Add some realistic noise
        queue_depth = max(0, int(cpu_util - 50 + random.uniform(-10, 20)))
        response_time = 200 + (cpu_util - 50) * 10 + random.uniform(-50, 100)
        response_time = max(50, response_time)  # Minimum 50ms
        
        # Create metrics
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=cpu_util,
            memory_utilization=memory_util,
            queue_depth=queue_depth,
            response_time_p95=response_time,
            carbon_intensity=carbon_intensity,
            active_requests=random.randint(10, 100)
        )
        
        # Update scaler with metrics
        await scaler.update_metrics(metrics)
        
        # Get current status
        status = scaler.get_scaling_status()
        
        print(f"   📈 CPU: {cpu_util:.1f}%, Memory: {memory_util:.1f}%, "
              f"Carbon: {carbon_intensity:.1f} gCO2/kWh")
        print(f"   🔧 Instances: {status['current_instances']}, "
              f"Queue: {queue_depth}, Response: {response_time:.0f}ms")
        
        if status['recent_actions']:
            last_action = status['recent_actions'][-1]
            if last_action['timestamp']:
                print(f"   📋 Last action: {last_action['direction']} "
                      f"(triggered by: {', '.join(last_action['triggered_by'])})")
        
        # Brief pause to simulate real-time
        await asyncio.sleep(0.1)
        
    # Show final scaling summary
    print(f"\n📊 Scaling Summary:")
    final_status = scaler.get_scaling_status()
    
    print(f"   Final instances: {final_status['current_instances']}")
    print(f"   Total scaling actions: {final_status['total_scaling_actions']}")
    print(f"   Carbon efficiency score: {final_status['carbon_efficiency_score']:.2f}")
    
    if final_status['usage_patterns'].get('status') != 'insufficient_data':
        patterns = final_status['usage_patterns']
        print(f"   Usage patterns detected:")
        print(f"     Peak hour: {patterns.get('peak_hour', 'N/A')} "
              f"({patterns.get('peak_cpu', 0):.1f}% CPU)")
        print(f"     Low hour: {patterns.get('low_hour', 'N/A')} "
              f"({patterns.get('low_cpu', 0):.1f}% CPU)")
        print(f"     Pattern strength: {patterns.get('pattern_strength', 'unknown')}")
    
    print(f"   Recent scaling actions:")
    for action in final_status['recent_actions'][-3:]:  # Last 3 actions
        print(f"     {action['direction'].upper()} to {action['target']} instances "
              f"(confidence: {action['confidence']:.2f})")


async def demo_carbon_aware_optimization():
    """Demo carbon-aware performance optimization."""
    print("\n" + "="*60)
    print("🌱 Carbon-Aware Performance Optimization Demo")
    print("="*60)
    
    # Create scaler with high carbon weight
    carbon_scaler = IntelligentAutoScaler(
        enable_carbon_aware=True,
        carbon_weight=0.7  # High carbon weight
    )
    
    print("🌍 Testing scaling decisions with different carbon scenarios...")
    
    # Test different carbon scenarios
    scenarios = [
        {"name": "Clean Energy", "carbon": 40, "expected": "scale_up"},
        {"name": "Moderate Carbon", "carbon": 150, "expected": "stable"},
        {"name": "High Carbon", "carbon": 350, "expected": "scale_down"},
        {"name": "Very High Carbon", "carbon": 500, "expected": "scale_down"}
    ]
    
    for scenario in scenarios:
        print(f"\n🧪 Scenario: {scenario['name']} ({scenario['carbon']} gCO2/kWh)")
        
        # Create metrics for this scenario
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=60,  # Moderate CPU usage
            memory_utilization=50,  # Moderate memory usage
            carbon_intensity=scenario['carbon'],
            queue_depth=20,
            response_time_p95=300,
            active_requests=50
        )
        
        # Update scaler
        await carbon_scaler.update_metrics(metrics)
        
        # Get status
        status = carbon_scaler.get_scaling_status()
        
        print(f"   Current instances: {status['current_instances']}")
        print(f"   Carbon efficiency score: {status['carbon_efficiency_score']:.2f}")
        
        if status['recent_actions']:
            last_action = status['recent_actions'][-1]
            triggered_by = last_action.get('triggered_by', [])
            if 'carbon_aware' in triggered_by:
                print(f"   🌱 Carbon-aware scaling triggered: {last_action['direction']}")
            else:
                print(f"   📊 Standard scaling triggered: {last_action['direction']}")
                
        await asyncio.sleep(0.5)  # Small delay between scenarios
    
    print(f"\n📈 Carbon Impact Analysis:")
    final_status = carbon_scaler.get_scaling_status()
    carbon_actions = [
        action for action in final_status['recent_actions']
        if 'carbon_aware' in action.get('triggered_by', [])
    ]
    
    print(f"   Total actions: {len(final_status['recent_actions'])}")
    print(f"   Carbon-triggered actions: {len(carbon_actions)}")
    print(f"   Final carbon efficiency: {final_status['carbon_efficiency_score']:.2f}")


async def demo_predictive_scaling():
    """Demo predictive scaling capabilities."""
    print("\n" + "="*60) 
    print("🔮 Predictive Scaling Demo")
    print("="*60)
    
    scaler = IntelligentAutoScaler(enable_predictive=True)
    
    print("📊 Building usage pattern history...")
    
    # Generate 24 hours of historical data with daily patterns
    for hour in range(24):
        for minute_offset in [0, 15, 30, 45]:  # 4 samples per hour
            # Simulate daily usage pattern
            # Higher usage during business hours (9 AM - 5 PM)
            if 9 <= hour <= 17:
                base_cpu = 70 + random.uniform(-10, 15)
            else:
                base_cpu = 30 + random.uniform(-5, 10)
                
            # Add weekly pattern (weekends lower)
            day_of_week = 2  # Tuesday (0 = Monday)
            if day_of_week >= 5:  # Weekend
                base_cpu *= 0.6
                
            cpu_util = max(10, min(95, base_cpu))
            memory_util = cpu_util * 0.8 + random.uniform(-10, 10)
            memory_util = max(20, min(90, memory_util))
            
            metrics = ScalingMetrics(
                timestamp=datetime.now() - timedelta(hours=23-hour, minutes=45-minute_offset),
                cpu_utilization=cpu_util,
                memory_utilization=memory_util,
                queue_depth=max(0, int((cpu_util - 50) / 5)),
                response_time_p95=200 + (cpu_util - 50) * 8,
                active_requests=int(cpu_util * 2)
            )
            
            await scaler.update_metrics(metrics)
            
    print("✅ Generated 24 hours of usage history")
    
    # Analyze patterns
    status = scaler.get_scaling_status()
    patterns = status.get('usage_patterns', {})
    
    if patterns.get('status') != 'insufficient_data':
        print(f"\n🎯 Detected Usage Patterns:")
        print(f"   Peak usage hour: {patterns.get('peak_hour', 'N/A')} "
              f"({patterns.get('peak_cpu', 0):.1f}% CPU)")
        print(f"   Low usage hour: {patterns.get('low_hour', 'N/A')} "
              f"({patterns.get('low_cpu', 0):.1f}% CPU)")
        print(f"   Usage variance: {patterns.get('cpu_variance', 0):.1f}")
        print(f"   Pattern strength: {patterns.get('pattern_strength', 'unknown')}")
        
        # Test prediction
        print(f"\n🔮 Testing demand prediction...")
        
        if hasattr(scaler.predictive_scaler, 'predict_demand'):
            prediction = scaler.predictive_scaler.predict_demand(30)  # Next 30 minutes
            if prediction is not None:
                print(f"   Predicted CPU usage (next 30min): {prediction:.1f}%")
                
                if prediction > 80:
                    print(f"   🚀 High demand predicted - recommend proactive scale-up")
                elif prediction < 30:
                    print(f"   📉 Low demand predicted - recommend scale-down")
                else:
                    print(f"   📊 Moderate demand predicted - maintain current scale")
            else:
                print(f"   ⚠️ Insufficient data for prediction")
    else:
        print(f"   ⚠️ Need more historical data to detect patterns")
    
    # Simulate real-time prediction accuracy
    print(f"\n📈 Simulating real-time predictions...")
    for i in range(5):
        # Simulate current high load
        current_cpu = 85 + random.uniform(-5, 10)
        
        metrics = ScalingMetrics(
            timestamp=datetime.now(),
            cpu_utilization=current_cpu,
            memory_utilization=current_cpu * 0.9,
            queue_depth=int((current_cpu - 50) / 3),
            response_time_p95=300 + (current_cpu - 50) * 10,
            active_requests=int(current_cpu * 2.5)
        )
        
        await scaler.update_metrics(metrics)
        
        # Check if predictive scaling triggered
        status = scaler.get_scaling_status()
        recent_actions = status.get('recent_actions', [])
        
        print(f"   Step {i+1}: CPU {current_cpu:.1f}%, "
              f"Instances: {status['current_instances']}")
        
        if recent_actions:
            last_action = recent_actions[-1]
            if 'predictive' in ', '.join(last_action.get('triggered_by', [])):
                print(f"     🔮 Predictive scaling activated!")
                
        await asyncio.sleep(0.2)


async def main():
    """Run all Generation 3 scaling and optimization demos."""
    print("🚀 Carbon-Aware-Trainer Generation 3 Scaling & Optimization Demo")
    print("This demo showcases the advanced scaling and performance capabilities.")
    
    try:
        # Run all demos
        await demo_performance_optimization()
        await demo_intelligent_caching()
        await demo_intelligent_scaling()
        await demo_carbon_aware_optimization()
        await demo_predictive_scaling()
        
        print("\n" + "="*60)
        print("🎉 Generation 3 Scaling & Optimization Demo Completed!")
        print("All advanced scaling and performance features are operational.")
        print("The system can now intelligently scale based on:")
        print("  • Resource utilization patterns")
        print("  • Carbon intensity forecasts")
        print("  • Predictive demand analysis")
        print("  • Performance optimization strategies")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo encountered an error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    # Run the demo
    result = asyncio.run(main())
    exit(result)