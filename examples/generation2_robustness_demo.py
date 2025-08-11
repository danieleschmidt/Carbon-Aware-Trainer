#!/usr/bin/env python3
"""
Generation 2 Robustness Features Demo

Demonstrates the robust reliability features:
- Circuit breaker pattern for resilient operations
- Comprehensive validation system
- Health monitoring and alerting
- Error handling and recovery
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta

from carbon_aware_trainer.core.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, circuit_breaker_manager
)
from carbon_aware_trainer.core.comprehensive_validation import (
    ComprehensiveValidator, ValidationSeverity, validator
)
from carbon_aware_trainer.core.health_monitoring import (
    HealthMonitor, AlertSeverity, health_monitor
)
from carbon_aware_trainer.core.types import CarbonIntensity, CarbonForecast, TrainingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_circuit_breaker():
    """Demo circuit breaker pattern for resilient operations."""
    print("\n" + "="*60)
    print("🔄 Circuit Breaker Pattern Demo")
    print("="*60)
    
    # Create circuit breaker with custom config
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=5.0,  # 5 seconds for demo
        success_threshold=2,
        timeout=2.0
    )
    
    breaker = CircuitBreaker("demo_service", config)
    
    # Simulate unreliable service
    call_count = 0
    
    async def unreliable_service():
        nonlocal call_count
        call_count += 1
        
        # Fail first 4 calls, then succeed
        if call_count <= 4:
            await asyncio.sleep(0.5)
            raise Exception(f"Service unavailable (call {call_count})")
        
        await asyncio.sleep(0.5)
        return f"Success on call {call_count}"
    
    print("🧪 Testing circuit breaker with failing service...")
    
    # Test circuit breaker behavior
    for i in range(10):
        try:
            result = await breaker.call(unreliable_service)
            print(f"✅ Call {i+1}: {result}")
        except CircuitBreakerError as e:
            print(f"⛔ Call {i+1}: Circuit OPEN - {e}")
        except Exception as e:
            print(f"❌ Call {i+1}: Service failed - {e}")
            
        # Show circuit breaker stats
        stats = breaker.get_stats()
        print(f"   State: {stats['state']}, Failures: {stats['failure_count']}")
        
        await asyncio.sleep(1)
    
    # Show final statistics
    final_stats = breaker.get_stats()
    print(f"\n📊 Final Circuit Breaker Stats:")
    print(f"   State: {final_stats['state']}")
    print(f"   Success Rate: {final_stats['success_rate_window']:.2%}")
    print(f"   Total Calls: {final_stats['total_calls_window']}")
    
    # Test circuit breaker manager
    print(f"\n🔧 Circuit Breaker Manager:")
    manager_stats = circuit_breaker_manager.get_all_stats()
    health_summary = circuit_breaker_manager.get_health_summary()
    print(f"   Health Status: {health_summary['status']}")
    print(f"   Total Breakers: {health_summary['total_breakers']}")


async def demo_comprehensive_validation():
    """Demo comprehensive validation system."""
    print("\n" + "="*60)
    print("🔍 Comprehensive Validation Demo")
    print("="*60)
    
    # Test carbon data validation
    print("🌍 Testing carbon data validation...")
    
    # Valid carbon intensity
    valid_carbon = CarbonIntensity(
        carbon_intensity=150.5,
        timestamp=datetime.now(),
        region="US-CA",
        confidence=0.9
    )
    
    result = validator.validate_carbon_data(valid_carbon)
    print(f"✅ Valid carbon data: {result.is_valid}")
    
    # Invalid carbon intensity
    invalid_carbon = CarbonIntensity(
        carbon_intensity=-50.0,  # Invalid negative value
        timestamp=datetime.now() - timedelta(hours=5),  # Old data
        region="",  # Missing region
        confidence=0.9
    )
    
    result = validator.validate_carbon_data(invalid_carbon)
    print(f"❌ Invalid carbon data: {result.is_valid}")
    
    if result.issues:
        print("   Issues found:")
        for issue in result.issues:
            severity_emoji = {
                "critical": "🔥",
                "error": "❌", 
                "warning": "⚠️",
                "info": "ℹ️"
            }
            print(f"   {severity_emoji.get(issue.severity.value, '?')} {issue.field}: {issue.message}")
            if issue.suggestion:
                print(f"      💡 Suggestion: {issue.suggestion}")
    
    # Test security validation
    print(f"\n🔒 Testing security validation...")
    
    # Safe input
    safe_data = {"region": "US-CA", "model_name": "bert-base"}
    result = validator.validate_api_input(safe_data)
    print(f"✅ Safe input: {result.is_valid}")
    
    # Malicious input
    malicious_data = {
        "region": "US-CA",
        "script": "<script>alert('xss')</script>",
        "path": "../../../etc/passwd",
        "command": "rm -rf /"
    }
    
    result = validator.validate_api_input(malicious_data)
    print(f"🚨 Malicious input detected: {result.is_valid}")
    
    if result.issues:
        critical_issues = result.get_issues_by_severity(ValidationSeverity.CRITICAL)
        print(f"   Critical security issues: {len(critical_issues)}")
        for issue in critical_issues:
            print(f"   🔥 {issue.field}: {issue.message}")
    
    # Test comprehensive validation
    print(f"\n🧪 Running comprehensive validation...")
    
    validation_data = {
        "carbon_data": valid_carbon,
        "memory_gb": 64,
        "gpu_count": 8,
        "user_input": "normal training data",
        "malicious_input": "<script>evil()</script>"
    }
    
    result = validator.run_comprehensive_validation(**validation_data)
    summary = validator.get_validation_summary(result)
    
    print(f"📋 Validation Summary:")
    print(f"   Overall Valid: {summary['is_valid']}")
    print(f"   Total Issues: {summary['total_issues']}")
    print(f"   Issues by Severity: {summary['issues_by_severity']}")
    
    if summary['critical_issues']:
        print(f"   🔥 Critical Issues:")
        for issue in summary['critical_issues']:
            print(f"      - {issue}")


async def demo_health_monitoring():
    """Demo health monitoring and alerting."""
    print("\n" + "="*60)
    print("💚 Health Monitoring & Alerting Demo")
    print("="*60)
    
    # Create custom alert handler
    def alert_handler(alert):
        severity_emoji = {
            "info": "ℹ️",
            "warning": "⚠️", 
            "error": "❌",
            "critical": "🔥"
        }
        emoji = severity_emoji.get(alert.severity.value, "?")
        print(f"   🚨 ALERT {emoji} {alert.component}: {alert.message}")
    
    health_monitor.add_alert_handler(alert_handler)
    
    # Add custom health check
    async def check_custom_service():
        # Simulate service that becomes unhealthy
        import random
        if random.random() < 0.3:  # 30% chance of failure
            raise Exception("Custom service is down")
        return "healthy"
    
    health_monitor.register_health_check(
        "custom_service",
        check_custom_service,
        interval=5.0,  # Check every 5 seconds
        critical=False
    )
    
    print("🏃‍♂️ Starting health monitoring...")
    await health_monitor.start_monitoring()
    
    # Let monitoring run for a bit
    print("⏱️  Running health checks for 15 seconds...")
    await asyncio.sleep(15)
    
    # Get health status
    health_status = await health_monitor.get_health_status()
    
    print(f"\n📊 System Health Status:")
    print(f"   Overall Status: {health_status['overall_status'].upper()}")
    print(f"   Active Alerts: {health_status['active_alerts']}")
    print(f"   Total Alerts: {health_status['total_alerts']}")
    
    print(f"\n🔍 Health Check Details:")
    for name, check in health_status['health_checks'].items():
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "❌",
            "unknown": "❓"
        }
        emoji = status_emoji.get(check['status'], "?")
        print(f"   {emoji} {name}: {check['status']} (failures: {check['failure_count']})")
    
    # Show system metrics
    metrics = health_status['system_metrics']
    print(f"\n📈 System Metrics:")
    if 'cpu' in metrics and 'error' not in metrics['cpu']:
        print(f"   CPU Usage: {metrics['cpu']['cpu_percent']:.1f}%")
    if 'memory' in metrics and 'error' not in metrics['memory']:
        print(f"   Memory Usage: {metrics['memory']['memory_percent']:.1f}%")
    if 'disk' in metrics and 'error' not in metrics['disk']:
        print(f"   Disk Usage: {metrics['disk']['disk_percent']:.1f}%")
    
    # Show recent alerts
    recent_alerts = health_monitor.get_alerts(since_hours=1)
    if recent_alerts:
        print(f"\n📢 Recent Alerts:")
        for alert in recent_alerts[-5:]:  # Last 5 alerts
            status = "✅ RESOLVED" if alert['resolved'] else "🔴 ACTIVE"
            print(f"   [{alert['timestamp'][:19]}] {status} {alert['severity'].upper()}: {alert['message']}")
    
    await health_monitor.stop_monitoring()


async def demo_error_recovery():
    """Demo error handling and recovery scenarios."""
    print("\n" + "="*60)
    print("🔧 Error Handling & Recovery Demo")
    print("="*60)
    
    # Simulate various error scenarios and recovery
    
    print("🧪 Testing graceful degradation...")
    
    # Test cascade failures with circuit breakers
    services = ["carbon_api", "training_service", "monitoring"]
    
    for service_name in services:
        print(f"\n💥 Simulating {service_name} failure...")
        
        breaker = circuit_breaker_manager.get_breaker(service_name)
        
        # Simulate service failures
        failure_count = 0
        
        async def failing_service():
            nonlocal failure_count
            failure_count += 1
            if failure_count < 6:  # Fail 5 times, then recover
                raise Exception(f"{service_name} connection failed")
            return f"{service_name} recovered"
        
        # Test service with circuit breaker
        for attempt in range(8):
            try:
                result = await breaker.call(failing_service)
                print(f"   ✅ Attempt {attempt + 1}: {result}")
            except CircuitBreakerError as e:
                print(f"   ⛔ Attempt {attempt + 1}: {e.service_name} circuit is OPEN")
            except Exception as e:
                print(f"   ❌ Attempt {attempt + 1}: {e}")
                
            await asyncio.sleep(0.5)
    
    # Show final system health
    print(f"\n📊 Final System Health:")
    manager_health = circuit_breaker_manager.get_health_summary()
    print(f"   Status: {manager_health['status']}")
    print(f"   Open Breakers: {manager_health['open_breakers']}")
    print(f"   Recovering Breakers: {manager_health['half_open_breakers']}")
    
    if manager_health['open_breaker_names']:
        print(f"   Services Down: {', '.join(manager_health['open_breaker_names'])}")
    
    # Test recovery
    if manager_health['status'] != 'healthy':
        print(f"\n🔄 Testing automatic recovery...")
        await asyncio.sleep(6)  # Wait for recovery timeout
        
        # Try one more time
        for service_name in manager_health['open_breaker_names']:
            breaker = circuit_breaker_manager.get_breaker(service_name)
            try:
                result = await breaker.call(lambda: asyncio.sleep(0.1) or f"{service_name} healthy")
                print(f"   ✅ {service_name} recovered successfully")
            except Exception as e:
                print(f"   ❌ {service_name} still failing: {e}")


async def main():
    """Run all Generation 2 robustness demos."""
    print("🛡️ Carbon-Aware-Trainer Generation 2 Robustness Demo")
    print("This demo showcases the enhanced reliability and monitoring capabilities.")
    
    try:
        # Run all demos
        await demo_circuit_breaker()
        await demo_comprehensive_validation()
        await demo_health_monitoring()
        await demo_error_recovery()
        
        print("\n" + "="*60)
        print("🎉 Generation 2 Robustness Demo Completed!")
        print("All reliability features are working and protecting the system.")
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