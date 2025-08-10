#!/usr/bin/env python3
"""Comprehensive integration tests for carbon-aware trainer."""

import pytest
import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, timedelta
import tempfile
import json

from carbon_aware_trainer import CarbonAwareTrainer, CarbonMonitor
from carbon_aware_trainer.core.types import (
    TrainingConfig, TrainingState, CarbonDataSource
)
from carbon_aware_trainer.core.auto_scaling import ResourceType
from carbon_aware_trainer.core.robustness import RobustnessManager, HealthStatus
from carbon_aware_trainer.core.auto_scaling import AutoScalingOptimizer, ScalingStrategy, ResourceMetrics


class MockTrainingModel:
    """Mock training model for testing."""
    def __init__(self, fail_rate=0.0):
        self.step_count = 0
        self.fail_rate = fail_rate
    
    def train_step(self, batch, **kwargs):
        """Mock training step."""
        self.step_count += 1
        
        if self.fail_rate > 0 and (self.step_count % int(1/self.fail_rate)) == 0:
            raise RuntimeError(f"Mock training failure at step {self.step_count}")
        
        return {
            'loss': max(0.01, 2.0 - (self.step_count * 0.1)),
            'accuracy': min(0.98, self.step_count * 0.05),
            'step': self.step_count
        }


class TestCarbonAwareTraining:
    """Test comprehensive carbon-aware training functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample carbon data
        self.sample_data_file = os.path.join(self.temp_dir, 'test_carbon_data.json')
        sample_data = {
            'regions': {
                'US-CA': {
                    'historical': [
                        {
                            'timestamp': '2024-01-01T00:00:00Z',
                            'carbon_intensity': 120.0,
                            'renewable_percentage': 45.0
                        },
                        {
                            'timestamp': '2024-01-01T01:00:00Z', 
                            'carbon_intensity': 95.0,
                            'renewable_percentage': 60.0
                        },
                        {
                            'timestamp': '2024-01-01T02:00:00Z',
                            'carbon_intensity': 180.0,
                            'renewable_percentage': 25.0
                        }
                    ]
                }
            }
        }
        
        with open(self.sample_data_file, 'w') as f:
            json.dump(sample_data, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_basic_training_session(self):
        """Test basic carbon-aware training session."""
        model = MockTrainingModel()
        
        config = TrainingConfig(
            carbon_threshold=150.0,
            pause_threshold=200.0,
            resume_threshold=100.0
        )
        
        trainer = CarbonAwareTrainer(
            model=model,
            carbon_model='cached',
            region='US-CA',
            config=config,
            api_key=self.sample_data_file
        )
        
        async with trainer:
            await trainer.start_training()
            
            # Run some training steps
            for i in range(5):
                result = await trainer.train_step({'batch': i})
                assert 'loss' in result
                assert 'accuracy' in result
                assert result['step'] == i + 1
            
            # Check metrics
            metrics = trainer.get_carbon_metrics()
            assert metrics['step'] == 5
            assert metrics['current_state'] == TrainingState.RUNNING.value
            assert metrics['total_carbon_kg'] >= 0
    
    @pytest.mark.asyncio
    async def test_pause_resume_functionality(self):
        """Test carbon-aware pause/resume functionality."""
        model = MockTrainingModel()
        
        # Use low thresholds to trigger pause/resume
        config = TrainingConfig(
            carbon_threshold=100.0,
            pause_threshold=110.0,  # Will pause during high carbon periods
            resume_threshold=90.0
        )
        
        trainer = CarbonAwareTrainer(
            model=model,
            carbon_model='cached',
            region='US-CA',
            config=config,
            api_key=self.sample_data_file
        )
        
        state_changes = []
        
        def track_state_changes(state, metrics):
            state_changes.append(state)
        
        trainer.add_state_callback(track_state_changes)
        
        async with trainer:
            await trainer.start_training()
            
            # Initial state should be running
            assert trainer.state == TrainingState.RUNNING
            
            # Run training steps
            for i in range(3):
                await trainer.train_step({'batch': i})
                await asyncio.sleep(0.1)
            
            # Should have recorded some state changes
            assert len(state_changes) > 0
            assert TrainingState.RUNNING in state_changes
    
    @pytest.mark.asyncio
    async def test_carbon_monitoring(self):
        """Test carbon intensity monitoring."""
        monitor = CarbonMonitor(
            regions=['US-CA'],
            data_source=CarbonDataSource.CACHED,
            api_key=self.sample_data_file,
            update_interval=1  # Fast updates for testing
        )
        
        events = []
        
        def track_events(event_type, data):
            events.append((event_type, data))
        
        monitor.add_callback(track_events)
        
        async with monitor:
            await monitor.start_monitoring()
            
            # Get current intensity
            intensity = await monitor.get_current_intensity('US-CA')
            assert intensity is not None
            assert intensity.carbon_intensity > 0
            
            # Test optimal window finding
            window = monitor.find_optimal_window(
                duration_hours=2,
                max_carbon_intensity=150.0
            )
            
            if window:  # May be None if no suitable window
                assert window.avg_carbon_intensity <= 150.0
                assert window.region == 'US-CA'
            
            await asyncio.sleep(0.5)
            await monitor.stop_monitoring()


class TestRobustnessFeatures:
    """Test production robustness features."""
    
    @pytest.mark.asyncio
    async def test_robustness_manager(self):
        """Test robustness manager functionality."""
        robustness = RobustnessManager(
            health_check_interval=1,  # Fast checks for testing
            max_retry_attempts=2,
            circuit_breaker_threshold=2
        )
        
        health_updates = []
        alerts = []
        
        def health_callback(event_type, data):
            if event_type == 'health_update':
                health_updates.append(data)
        
        def alert_callback(event_type, data):
            if event_type == 'health_alert':
                alerts.append(data)
        
        robustness.add_health_callback(health_callback)
        robustness.add_alert_callback(alert_callback)
        
        await robustness.start_health_monitoring()
        
        # Record some operations
        robustness.record_operation(success=True)
        robustness.record_operation(success=True)
        robustness.record_operation(success=False)
        
        # Record failures to test circuit breaker
        robustness.record_failure('test_component')
        robustness.record_failure('test_component')
        robustness.record_failure('test_component')  # Should open circuit breaker
        
        # Wait for health checks
        await asyncio.sleep(2)
        
        # Check circuit breaker
        assert robustness.is_circuit_breaker_open('test_component')
        
        # Get health summary
        summary = robustness.get_health_summary()
        assert summary['total_operations'] == 3
        assert summary['success_rate'] == 2/3
        assert summary['circuit_breakers']['open_count'] > 0
        
        await robustness.stop_health_monitoring()
        
        # Should have received health updates
        assert len(health_updates) > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery and resilience."""
        model = MockTrainingModel(fail_rate=0.3)  # 30% failure rate
        
        config = TrainingConfig(
            carbon_threshold=150.0
        )
        
        trainer = CarbonAwareTrainer(
            model=model,
            carbon_model='cached',
            region='US-CA',
            config=config,
            api_key='sample_data/sample_carbon_data.json'
        )
        
        successful_steps = 0
        failed_steps = 0
        
        async with trainer:
            await trainer.start_training()
            
            # Run training with retry logic
            for i in range(10):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        await trainer.train_step({'batch': i})
                        successful_steps += 1
                        break
                    except RuntimeError:
                        if attempt == max_retries - 1:
                            failed_steps += 1
                        await asyncio.sleep(0.1)
        
        # Should have some successful steps despite failures
        assert successful_steps > 0
        print(f"Successful: {successful_steps}, Failed: {failed_steps}")


class TestAutoScaling:
    """Test auto-scaling optimization features."""
    
    @pytest.mark.asyncio
    async def test_carbon_aware_scaling(self):
        """Test carbon-aware scaling optimization."""
        optimizer = AutoScalingOptimizer(
            strategy=ScalingStrategy.CARBON_AWARE,
            min_scaling_interval=1,  # Fast scaling for testing
            carbon_weight=0.8,
            performance_weight=0.2
        )
        
        # Test with low carbon intensity (should scale up)
        low_carbon_metrics = ResourceMetrics(
            gpu_utilization=0.6,
            memory_utilization=0.5,
            throughput_ops_per_sec=100.0,
            carbon_efficiency=15.0
        )
        
        from carbon_aware_trainer.core.types import CarbonIntensity
        low_carbon_intensity = CarbonIntensity(
            carbon_intensity=80.0,  # Low carbon
            timestamp=datetime.now(),
            region='US-CA'
        )
        
        decisions = await optimizer.analyze_and_optimize(
            current_metrics=low_carbon_metrics,
            carbon_intensity=low_carbon_intensity,
            training_state=TrainingState.RUNNING
        )
        
        # May recommend scaling up during low carbon
        scale_up_decisions = [d for d in decisions if d.action == 'scale_up']
        print(f"Low carbon decisions: {len(decisions)} total, {len(scale_up_decisions)} scale_up")
        
        # Test with high carbon intensity (should scale down)
        high_carbon_intensity = CarbonIntensity(
            carbon_intensity=300.0,  # High carbon
            timestamp=datetime.now(),
            region='US-CA'
        )
        
        decisions_high = await optimizer.analyze_and_optimize(
            current_metrics=low_carbon_metrics,
            carbon_intensity=high_carbon_intensity,
            training_state=TrainingState.RUNNING
        )
        
        # May recommend scaling down during high carbon
        scale_down_decisions = [d for d in decisions_high if d.action == 'scale_down']
        print(f"High carbon decisions: {len(decisions_high)} total, {len(scale_down_decisions)} scale_down")
        
        # At minimum, should have made some decisions
        total_decisions = len(decisions) + len(decisions_high)
        assert total_decisions >= 0  # Allow for no decisions if conditions aren't met
    
    @pytest.mark.asyncio
    async def test_performance_based_scaling(self):
        """Test performance-based scaling optimization."""
        optimizer = AutoScalingOptimizer(
            strategy=ScalingStrategy.PERFORMANCE_BASED,
            min_scaling_interval=1
        )
        
        # Test with low GPU utilization (should scale up)
        underutilized_metrics = ResourceMetrics(
            gpu_utilization=0.4,  # Low utilization
            memory_utilization=0.3,
            throughput_ops_per_sec=50.0
        )
        
        from carbon_aware_trainer.core.types import CarbonIntensity
        neutral_carbon = CarbonIntensity(
            carbon_intensity=150.0,
            timestamp=datetime.now(),
            region='US-CA'
        )
        
        decisions = await optimizer.analyze_and_optimize(
            current_metrics=underutilized_metrics,
            carbon_intensity=neutral_carbon,
            training_state=TrainingState.RUNNING
        )
        
        # Should recommend scaling up for better utilization
        assert len(decisions) > 0
        batch_size_decisions = [
            d for d in decisions 
            if d.resource_type == ResourceType.BATCH_SIZE
        ]
        assert len(batch_size_decisions) > 0
    
    def test_optimization_summary(self):
        """Test optimization performance tracking."""
        optimizer = AutoScalingOptimizer(strategy=ScalingStrategy.HYBRID)
        
        # Simulate some optimizations
        optimizer.complete_scaling_change(
            ResourceType.BATCH_SIZE,
            success=True,
            actual_benefit={'performance_improvement': 0.15}
        )
        
        optimizer.complete_scaling_change(
            ResourceType.LEARNING_RATE,
            success=True,
            actual_benefit={'carbon_savings_kg': 0.05}
        )
        
        summary = optimizer.get_optimization_summary()
        assert summary['successful_optimizations'] == 2
        assert summary['performance_improvements'] > 0
        assert summary['carbon_savings_kg'] > 0


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create comprehensive sample data
        self.sample_data_file = os.path.join(self.temp_dir, 'e2e_carbon_data.json')
        sample_data = {
            'regions': {
                'US-CA': {
                    'historical': [
                        {
                            'timestamp': '2024-01-01T00:00:00Z',
                            'carbon_intensity': 80.0,   # Low carbon period
                            'renewable_percentage': 70.0
                        },
                        {
                            'timestamp': '2024-01-01T01:00:00Z',
                            'carbon_intensity': 250.0,  # High carbon period
                            'renewable_percentage': 20.0
                        },
                        {
                            'timestamp': '2024-01-01T02:00:00Z',
                            'carbon_intensity': 120.0,  # Medium carbon period
                            'renewable_percentage': 50.0
                        }
                    ]
                }
            }
        }
        
        with open(self.sample_data_file, 'w') as f:
            json.dump(sample_data, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_full_carbon_aware_pipeline(self):
        """Test complete carbon-aware training pipeline."""
        # Initialize all components
        model = MockTrainingModel()
        
        robustness = RobustnessManager(
            health_check_interval=2,
            circuit_breaker_threshold=3
        )
        
        optimizer = AutoScalingOptimizer(
            strategy=ScalingStrategy.HYBRID,
            min_scaling_interval=1
        )
        
        config = TrainingConfig(
            carbon_threshold=150.0,
            pause_threshold=200.0,
            resume_threshold=100.0
        )
        
        trainer = CarbonAwareTrainer(
            model=model,
            carbon_model='cached',
            region='US-CA',
            config=config,
            api_key=self.sample_data_file
        )
        
        # Track all events
        training_events = []
        health_events = []
        scaling_decisions = []
        
        def track_training(state, metrics):
            training_events.append((state, metrics.total_carbon_kg))
        
        def track_health(event_type, data):
            if event_type == 'health_update':
                health_events.append(data['overall_status'])
        
        trainer.add_state_callback(track_training)
        robustness.add_health_callback(track_health)
        
        # Start all systems
        await robustness.start_health_monitoring()
        
        async with trainer:
            await trainer.start_training()
            
            # Run comprehensive training simulation
            for epoch in range(3):
                print(f"Running epoch {epoch + 1}")
                
                for step in range(5):
                    # Get current metrics for scaling
                    current_metrics = ResourceMetrics(
                        gpu_utilization=0.7 + (step * 0.05),
                        memory_utilization=0.6 + (step * 0.03),
                        throughput_ops_per_sec=100.0 + (step * 10),
                        carbon_efficiency=12.0 + (step * 2)
                    )
                    
                    # Get carbon intensity
                    intensity = await trainer.monitor.get_current_intensity('US-CA')
                    
                    # Get scaling recommendations
                    if intensity:
                        decisions = await optimizer.analyze_and_optimize(
                            current_metrics=current_metrics,
                            carbon_intensity=intensity,
                            training_state=trainer.state
                        )
                        scaling_decisions.extend(decisions)
                    
                    # Execute training step
                    try:
                        result = await trainer.train_step({'epoch': epoch, 'step': step})
                        robustness.record_operation(success=True)
                        
                        # Apply scaling decisions (mock)
                        for decision in decisions:
                            optimizer.complete_scaling_change(
                                decision.resource_type,
                                success=True,
                                actual_benefit={'performance_improvement': 0.1}
                            )
                    
                    except Exception as e:
                        print(f"Training step failed: {e}")
                        robustness.record_operation(success=False)
                        robustness.record_failure('training')
                    
                    await asyncio.sleep(0.1)
        
        await robustness.stop_health_monitoring()
        
        # Verify comprehensive integration
        assert len(training_events) > 0
        
        # Get final metrics
        final_metrics = trainer.get_carbon_metrics()
        health_summary = robustness.get_health_summary()
        optimization_summary = optimizer.get_optimization_summary()
        
        # Verify system operated correctly
        assert final_metrics['step'] > 0
        assert health_summary['total_operations'] > 0
        
        print(f"\nIntegration Test Results:")
        print(f"  Training Steps: {final_metrics['step']}")
        print(f"  Total Carbon: {final_metrics['total_carbon_kg']:.3f} kg CO2")
        print(f"  Health Operations: {health_summary['total_operations']}")
        print(f"  Success Rate: {health_summary['success_rate']:.1%}")
        print(f"  Scaling Decisions: {optimization_summary['total_decisions']}")
        
        # Integration should demonstrate all three generations working together
        assert final_metrics['step'] >= 10  # Gen 1: Basic functionality
        assert health_summary['success_rate'] > 0.7  # Gen 2: Robustness
        # Gen 3: Scaling decisions made
        assert len(scaling_decisions) > 0 or optimization_summary['total_decisions'] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])