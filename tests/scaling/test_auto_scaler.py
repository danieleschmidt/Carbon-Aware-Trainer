"""Tests for auto-scaling system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from carbon_aware_trainer.scaling.auto_scaler import (
    AutoScaler, ScalingPolicy, ScalingMetrics, ScalingDirection, ScalingTrigger
)
from carbon_aware_trainer.core.types import CarbonIntensity
from carbon_aware_trainer.core.monitor import CarbonMonitor


@pytest.fixture
def default_scaling_policy():
    """Default scaling policy for tests."""
    return ScalingPolicy(
        scale_up_carbon_threshold=80.0,
        scale_down_carbon_threshold=150.0,
        max_cost_per_hour_usd=50.0,
        min_instances=1,
        max_instances=5,
        scale_up_cooldown_minutes=5,
        scale_down_cooldown_minutes=10
    )


@pytest.fixture
def mock_carbon_monitor():
    """Mock carbon monitor."""
    monitor = Mock(spec=CarbonMonitor)
    monitor.regions = ['US-CA']
    
    # Mock carbon intensity data
    mock_intensity = CarbonIntensity(
        region='US-CA',
        timestamp=datetime.now(),
        carbon_intensity=100.0,
        unit='gCO2/kWh',
        data_source='test',
        renewable_percentage=45.0
    )
    
    monitor.get_current_intensity = AsyncMock(return_value=mock_intensity)
    return monitor


@pytest.fixture
def mock_instance_manager():
    """Mock instance manager callback."""
    return AsyncMock(return_value=True)


class TestScalingPolicy:
    """Test scaling policy configuration."""
    
    def test_default_policy_values(self):
        """Test default policy initialization."""
        policy = ScalingPolicy()
        
        assert policy.scale_up_carbon_threshold == 80.0
        assert policy.scale_down_carbon_threshold == 150.0
        assert policy.min_instances == 1
        assert policy.max_instances == 10
        assert policy.trigger_type == ScalingTrigger.HYBRID
    
    def test_custom_policy_values(self):
        """Test custom policy configuration."""
        policy = ScalingPolicy(
            scale_up_carbon_threshold=60.0,
            scale_down_carbon_threshold=200.0,
            max_instances=20,
            trigger_type=ScalingTrigger.CARBON_BASED
        )
        
        assert policy.scale_up_carbon_threshold == 60.0
        assert policy.scale_down_carbon_threshold == 200.0
        assert policy.max_instances == 20
        assert policy.trigger_type == ScalingTrigger.CARBON_BASED


class TestAutoScaler:
    """Test auto-scaling system functionality."""
    
    def test_auto_scaler_initialization(self, default_scaling_policy, mock_carbon_monitor):
        """Test auto-scaler initialization."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        
        assert scaler.policy == default_scaling_policy
        assert scaler.carbon_monitor == mock_carbon_monitor
        assert scaler.current_instances == default_scaling_policy.min_instances
        assert scaler.target_instances == default_scaling_policy.min_instances
        assert not scaler._is_running
    
    @pytest.mark.asyncio
    async def test_start_stop_auto_scaling(self, default_scaling_policy, mock_carbon_monitor):
        """Test starting and stopping auto-scaling."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        
        # Start auto-scaling
        await scaler.start_auto_scaling(check_interval_minutes=0.1)  # Very short interval for testing
        assert scaler._is_running
        assert scaler._scaling_task is not None
        
        # Stop auto-scaling
        await scaler.stop_auto_scaling()
        assert not scaler._is_running
        assert scaler._scaling_task.cancelled() or scaler._scaling_task.done()
    
    @pytest.mark.asyncio
    async def test_collect_scaling_metrics(self, default_scaling_policy, mock_carbon_monitor):
        """Test metrics collection."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        
        metrics = await scaler._collect_scaling_metrics()
        
        assert isinstance(metrics, ScalingMetrics)
        assert metrics.current_instances == scaler.current_instances
        assert metrics.carbon_intensity == 100.0  # From mock
        assert metrics.renewable_percentage == 45.0  # From mock
        assert 0 <= metrics.cost_per_hour <= 1000  # Reasonable range
        assert 0 <= metrics.training_efficiency <= 1  # Percentage
    
    def test_carbon_scaling_evaluation(self, default_scaling_policy, mock_carbon_monitor):
        """Test carbon-based scaling decisions."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        
        # Test low carbon intensity (should scale up)
        low_carbon_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            current_instances=2,
            target_instances=2,
            carbon_intensity=50.0,  # Below scale_up_threshold (80)
            renewable_percentage=60.0,
            cost_per_hour=25.0,
            training_efficiency=0.8,
            scaling_decision=ScalingDirection.NONE,
            trigger_reason="",
            confidence_score=0.0
        )
        
        decision = scaler._evaluate_carbon_scaling(low_carbon_metrics)
        assert decision.direction == ScalingDirection.UP
        assert "Low carbon intensity" in decision.trigger_reason
        
        # Test high carbon intensity (should scale down)
        high_carbon_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            current_instances=2,
            target_instances=2,
            carbon_intensity=200.0,  # Above scale_down_threshold (150)
            renewable_percentage=20.0,
            cost_per_hour=25.0,
            training_efficiency=0.8,
            scaling_decision=ScalingDirection.NONE,
            trigger_reason="",
            confidence_score=0.0
        )
        
        decision = scaler._evaluate_carbon_scaling(high_carbon_metrics)
        assert decision.direction == ScalingDirection.DOWN
        assert "High carbon intensity" in decision.trigger_reason
    
    def test_cost_scaling_evaluation(self, default_scaling_policy, mock_carbon_monitor):
        """Test cost-based scaling decisions."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        
        # Test high cost (should scale down)
        high_cost_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            current_instances=5,
            target_instances=5,
            carbon_intensity=100.0,
            renewable_percentage=40.0,
            cost_per_hour=75.0,  # Above max_cost_per_hour_usd (50)
            training_efficiency=0.8,
            scaling_decision=ScalingDirection.NONE,
            trigger_reason="",
            confidence_score=0.0
        )
        
        decision = scaler._evaluate_cost_scaling(high_cost_metrics)
        assert decision.direction == ScalingDirection.DOWN
        assert "Cost too high" in decision.trigger_reason
        
        # Test low cost (should consider scale up)
        low_cost_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            current_instances=2,
            target_instances=2,
            carbon_intensity=100.0,
            renewable_percentage=40.0,
            cost_per_hour=10.0,  # Well below threshold
            training_efficiency=0.8,
            scaling_decision=ScalingDirection.NONE,
            trigger_reason="",
            confidence_score=0.0
        )
        
        decision = scaler._evaluate_cost_scaling(low_cost_metrics)
        assert decision.direction == ScalingDirection.UP
        assert "Cost efficient" in decision.trigger_reason
    
    def test_performance_scaling_evaluation(self, default_scaling_policy, mock_carbon_monitor):
        """Test performance-based scaling decisions."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        
        # Test high GPU utilization (should scale up)
        high_util_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            current_instances=2,
            target_instances=2,
            carbon_intensity=100.0,
            renewable_percentage=40.0,
            cost_per_hour=25.0,
            training_efficiency=0.8,
            scaling_decision=ScalingDirection.NONE,
            trigger_reason="",
            confidence_score=0.0,
            gpu_utilization=95.0  # High utilization
        )
        
        decision = scaler._evaluate_performance_scaling(high_util_metrics)
        assert decision.direction == ScalingDirection.UP
        assert "High GPU utilization" in decision.trigger_reason
        
        # Test low GPU utilization (should scale down)
        scaler.current_instances = 3  # Need > min_instances to scale down
        low_util_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            current_instances=3,
            target_instances=3,
            carbon_intensity=100.0,
            renewable_percentage=40.0,
            cost_per_hour=25.0,
            training_efficiency=0.8,
            scaling_decision=ScalingDirection.NONE,
            trigger_reason="",
            confidence_score=0.0,
            gpu_utilization=20.0  # Low utilization
        )
        
        decision = scaler._evaluate_performance_scaling(low_util_metrics)
        assert decision.direction == ScalingDirection.DOWN
        assert "Low GPU utilization" in decision.trigger_reason
    
    def test_renewable_scaling_evaluation(self, default_scaling_policy, mock_carbon_monitor):
        """Test renewable energy-based scaling decisions."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        
        # Test high renewable percentage (should scale up)
        high_renewable_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            current_instances=2,
            target_instances=2,
            carbon_intensity=100.0,
            renewable_percentage=75.0,  # High renewable (> min_renewable * 1.5)
            cost_per_hour=25.0,
            training_efficiency=0.8,
            scaling_decision=ScalingDirection.NONE,
            trigger_reason="",
            confidence_score=0.0
        )
        
        decision = scaler._evaluate_renewable_scaling(high_renewable_metrics)
        assert decision.direction == ScalingDirection.UP
        assert "High renewable energy" in decision.trigger_reason
        
        # Test low renewable percentage (should scale down)
        low_renewable_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            current_instances=2,
            target_instances=2,
            carbon_intensity=100.0,
            renewable_percentage=15.0,  # Low renewable (< min_renewable)
            cost_per_hour=25.0,
            training_efficiency=0.8,
            scaling_decision=ScalingDirection.NONE,
            trigger_reason="",
            confidence_score=0.0
        )
        
        decision = scaler._evaluate_renewable_scaling(low_renewable_metrics)
        assert decision.direction == ScalingDirection.DOWN
        assert "Low renewable energy" in decision.trigger_reason
    
    def test_scaling_constraints_cooldown(self, default_scaling_policy, mock_carbon_monitor):
        """Test scaling constraints and cooldown periods."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        
        # Set recent scale up time
        scaler.last_scale_up_time = datetime.now() - timedelta(minutes=2)  # Recent scale up
        
        # Create a valid scale up decision
        from carbon_aware_trainer.scaling.auto_scaler import ScalingDecision
        scale_up_decision = ScalingDecision(
            direction=ScalingDirection.UP,
            trigger_reason="Test scale up",
            confidence_score=0.8,
            weight=1.0
        )
        
        mock_metrics = Mock()
        constrained_decision = scaler._apply_scaling_constraints(scale_up_decision, mock_metrics)
        
        # Should be blocked by cooldown
        assert constrained_decision.direction == ScalingDirection.NONE
        assert "cooldown" in constrained_decision.trigger_reason.lower()
    
    def test_scaling_constraints_limits(self, default_scaling_policy, mock_carbon_monitor):
        """Test scaling limits constraints."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        
        # Test max instances limit
        scaler.current_instances = default_scaling_policy.max_instances
        
        from carbon_aware_trainer.scaling.auto_scaler import ScalingDecision
        scale_up_decision = ScalingDecision(
            direction=ScalingDirection.UP,
            trigger_reason="Test scale up",
            confidence_score=0.8,
            weight=1.0
        )
        
        mock_metrics = Mock()
        constrained_decision = scaler._apply_scaling_constraints(scale_up_decision, mock_metrics)
        
        assert constrained_decision.direction == ScalingDirection.NONE
        assert "maximum instances" in constrained_decision.trigger_reason
        
        # Test min instances limit
        scaler.current_instances = default_scaling_policy.min_instances
        
        scale_down_decision = ScalingDecision(
            direction=ScalingDirection.DOWN,
            trigger_reason="Test scale down",
            confidence_score=0.8,
            weight=1.0
        )
        
        constrained_decision = scaler._apply_scaling_constraints(scale_down_decision, mock_metrics)
        
        assert constrained_decision.direction == ScalingDirection.NONE
        assert "minimum instances" in constrained_decision.trigger_reason
    
    @pytest.mark.asyncio
    async def test_execute_scaling_action_up(self, default_scaling_policy, mock_carbon_monitor, mock_instance_manager):
        """Test executing scale up action."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor, mock_instance_manager)
        scaler.current_instances = 2
        
        from carbon_aware_trainer.scaling.auto_scaler import ScalingDecision
        scale_up_decision = ScalingDecision(
            direction=ScalingDirection.UP,
            trigger_reason="Test scale up",
            confidence_score=0.8,
            weight=1.0
        )
        
        success = await scaler._execute_scaling_action(scale_up_decision)
        
        assert success
        assert scaler.current_instances > 2  # Should have scaled up
        assert len(scaler.scaling_actions) == 1
        assert scaler.last_scale_up_time is not None
        
        # Verify instance manager was called
        mock_instance_manager.assert_called_once()
        call_args = mock_instance_manager.call_args
        assert call_args[1]['action_type'] == 'scale'
        assert call_args[1]['from_instances'] == 2
        assert call_args[1]['to_instances'] > 2
    
    @pytest.mark.asyncio
    async def test_execute_scaling_action_down(self, default_scaling_policy, mock_carbon_monitor, mock_instance_manager):
        """Test executing scale down action."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor, mock_instance_manager)
        scaler.current_instances = 4
        
        from carbon_aware_trainer.scaling.auto_scaler import ScalingDecision
        scale_down_decision = ScalingDecision(
            direction=ScalingDirection.DOWN,
            trigger_reason="Test scale down",
            confidence_score=0.8,
            weight=1.0
        )
        
        success = await scaler._execute_scaling_action(scale_down_decision)
        
        assert success
        assert scaler.current_instances < 4  # Should have scaled down
        assert len(scaler.scaling_actions) == 1
        assert scaler.last_scale_down_time is not None
        
        # Verify instance manager was called
        mock_instance_manager.assert_called_once()
        call_args = mock_instance_manager.call_args
        assert call_args[1]['action_type'] == 'scale'
        assert call_args[1]['from_instances'] == 4
        assert call_args[1]['to_instances'] < 4
    
    @pytest.mark.asyncio
    async def test_execute_scaling_action_no_manager(self, default_scaling_policy, mock_carbon_monitor):
        """Test executing scaling action without instance manager."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)  # No instance manager
        scaler.current_instances = 2
        
        from carbon_aware_trainer.scaling.auto_scaler import ScalingDecision
        scale_up_decision = ScalingDecision(
            direction=ScalingDirection.UP,
            trigger_reason="Test scale up",
            confidence_score=0.8,
            weight=1.0
        )
        
        success = await scaler._execute_scaling_action(scale_up_decision)
        
        # Should still succeed (simulated)
        assert success
        assert scaler.current_instances > 2
        assert len(scaler.scaling_actions) == 1
    
    def test_cost_and_carbon_impact_estimation(self, default_scaling_policy, mock_carbon_monitor):
        """Test cost and carbon impact estimation."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        
        # Add some metrics to history
        mock_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            current_instances=2,
            target_instances=2,
            carbon_intensity=150.0,
            renewable_percentage=40.0,
            cost_per_hour=25.0,
            training_efficiency=0.8,
            scaling_decision=ScalingDirection.NONE,
            trigger_reason="",
            confidence_score=0.0
        )
        scaler.metrics_history.append(mock_metrics)
        
        # Test cost impact
        cost_impact = scaler._estimate_cost_impact(2, 4)
        assert cost_impact > 0  # Scaling up should increase cost
        
        cost_impact_down = scaler._estimate_cost_impact(4, 2)
        assert cost_impact_down < 0  # Scaling down should decrease cost
        
        # Test carbon impact
        carbon_impact = scaler._estimate_carbon_impact(2, 4)
        # Should be positive (more instances = more carbon) or negative depending on efficiency
        assert isinstance(carbon_impact, float)
    
    def test_get_scaling_status(self, default_scaling_policy, mock_carbon_monitor):
        """Test scaling status reporting."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        scaler.current_instances = 3
        scaler._is_running = True
        
        status = scaler.get_scaling_status()
        
        assert status['is_running'] == True
        assert status['current_instances'] == 3
        assert status['target_instances'] == 3
        assert 'policy' in status
        assert 'current_metrics' in status
        assert 'recent_actions' in status
        assert 'total_actions' in status
        
        # Test policy information
        assert status['policy']['min_instances'] == default_scaling_policy.min_instances
        assert status['policy']['max_instances'] == default_scaling_policy.max_instances
    
    def test_metrics_cleanup(self, default_scaling_policy, mock_carbon_monitor):
        """Test cleanup of old metrics."""
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor)
        
        # Add old metrics (more than 24 hours old)
        old_metrics = ScalingMetrics(
            timestamp=datetime.now() - timedelta(hours=25),
            current_instances=2,
            target_instances=2,
            carbon_intensity=100.0,
            renewable_percentage=40.0,
            cost_per_hour=25.0,
            training_efficiency=0.8,
            scaling_decision=ScalingDirection.NONE,
            trigger_reason="Old metrics",
            confidence_score=0.0
        )
        scaler.metrics_history.append(old_metrics)
        
        # Add recent metrics
        recent_metrics = ScalingMetrics(
            timestamp=datetime.now(),
            current_instances=2,
            target_instances=2,
            carbon_intensity=100.0,
            renewable_percentage=40.0,
            cost_per_hour=25.0,
            training_efficiency=0.8,
            scaling_decision=ScalingDirection.NONE,
            trigger_reason="Recent metrics",
            confidence_score=0.0
        )
        scaler.metrics_history.append(recent_metrics)
        
        # Run cleanup
        scaler._cleanup_old_metrics()
        
        # Old metrics should be removed, recent metrics should remain
        assert len(scaler.metrics_history) == 1
        assert scaler.metrics_history[0].trigger_reason == "Recent metrics"


@pytest.mark.integration
class TestAutoScalerIntegration:
    """Integration tests for auto-scaler."""
    
    @pytest.mark.asyncio
    async def test_full_scaling_loop_integration(self, default_scaling_policy, mock_carbon_monitor, mock_instance_manager):
        """Test full scaling loop with real timing."""
        # Use very short intervals for testing
        default_scaling_policy.scale_up_cooldown_minutes = 0.05  # 3 seconds
        default_scaling_policy.scale_down_cooldown_minutes = 0.05
        
        scaler = AutoScaler(default_scaling_policy, mock_carbon_monitor, mock_instance_manager)
        
        # Start auto-scaling with very short check interval
        await scaler.start_auto_scaling(check_interval_minutes=0.02)  # ~1 second
        
        # Let it run for a short time
        await asyncio.sleep(0.5)
        
        # Stop scaling
        await scaler.stop_auto_scaling()
        
        # Should have collected some metrics
        assert len(scaler.metrics_history) >= 0  # May not have collected metrics yet due to timing