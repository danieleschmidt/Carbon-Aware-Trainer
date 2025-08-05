"""Unit tests for auto-scaling functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from carbon_aware_trainer.core.scaling import (
    CarbonAwareAutoScaler, ComputeNode, ResourceRequirement,
    ScalingDecision, ScalingEvent
)


class TestCarbonAwareAutoScaler:
    """Test cases for CarbonAwareAutoScaler."""
    
    @pytest.fixture
    def autoscaler(self):
        """Create autoscaler for testing."""
        return CarbonAwareAutoScaler(
            min_nodes=1,
            max_nodes=5,
            target_utilization=0.7,
            carbon_threshold=100.0,
            cooldown_minutes=1
        )
    
    @pytest.fixture
    def sample_nodes(self):
        """Create sample compute nodes."""
        return [
            ComputeNode(
                node_id="node1",
                region="US-CA",
                available_cpus=8,
                available_memory_gb=32,
                available_gpus=1,
                gpu_type="V100",
                cost_per_hour=2.5,
                utilization=0.8,
                carbon_intensity=80.0,
                last_updated=datetime.now()
            ),
            ComputeNode(
                node_id="node2",
                region="US-WA",
                available_cpus=8,
                available_memory_gb=32,
                available_gpus=1,
                gpu_type="V100",
                cost_per_hour=2.0,
                utilization=0.6,
                carbon_intensity=60.0,
                last_updated=datetime.now()
            )
        ]
    
    def test_node_management(self, autoscaler, sample_nodes):
        """Test adding and removing nodes."""
        # Test adding nodes
        for node in sample_nodes:
            autoscaler.add_node(node)
        
        assert len(autoscaler.current_nodes) == 2
        assert "node1" in autoscaler.current_nodes
        assert "node2" in autoscaler.current_nodes
        
        # Test removing node
        assert autoscaler.remove_node("node1") is True
        assert len(autoscaler.current_nodes) == 1
        assert "node1" not in autoscaler.current_nodes
        
        # Test removing non-existent node
        assert autoscaler.remove_node("nonexistent") is False
    
    def test_node_metrics_update(self, autoscaler, sample_nodes):
        """Test updating node metrics."""
        node = sample_nodes[0]
        autoscaler.add_node(node)
        
        # Update metrics
        autoscaler.update_node_metrics("node1", utilization=0.9, carbon_intensity=120.0)
        
        updated_node = autoscaler.current_nodes["node1"]
        assert updated_node.utilization == 0.9
        assert updated_node.carbon_intensity == 120.0
        assert updated_node.last_updated is not None
    
    def test_utilization_calculation(self, autoscaler, sample_nodes):
        """Test average utilization calculation."""
        for node in sample_nodes:
            autoscaler.add_node(node)
        
        avg_utilization = autoscaler._calculate_average_utilization()
        expected = (0.8 + 0.6) / 2
        assert avg_utilization == expected
    
    def test_carbon_intensity_calculation(self, autoscaler, sample_nodes):
        """Test average carbon intensity calculation."""
        for node in sample_nodes:
            autoscaler.add_node(node)
        
        # Add some carbon history
        now = datetime.now()
        autoscaler._carbon_history = [
            (now - timedelta(minutes=5), 80.0, "US-CA"),
            (now - timedelta(minutes=3), 60.0, "US-WA"),
            (now - timedelta(minutes=1), 90.0, "US-CA")
        ]
        
        avg_carbon = autoscaler._calculate_average_carbon_intensity()
        expected = (80.0 + 60.0 + 90.0) / 3
        assert avg_carbon == expected
    
    def test_cleanest_region_detection(self, autoscaler):
        """Test finding cleanest region."""
        # Add carbon history for different regions
        now = datetime.now()
        autoscaler._carbon_history = [
            (now, 120.0, "US-CA"),
            (now, 80.0, "US-WA"),
            (now, 60.0, "EU-FR")
        ]
        
        cleanest = autoscaler._find_cleanest_region()
        assert cleanest is not None
        assert cleanest[0] == "EU-FR"
        assert cleanest[1] == 60.0
    
    @pytest.mark.asyncio
    async def test_scale_up_evaluation(self, autoscaler, sample_nodes):
        """Test scale up decision evaluation."""
        for node in sample_nodes:
            autoscaler.add_node(node)
        
        # Set high utilization
        for node in autoscaler.current_nodes.values():
            node.utilization = 0.9  # Above scale_up_threshold (0.8)
        
        decision = await autoscaler.evaluate_scaling_decision()
        
        assert decision is not None
        assert decision.event == ScalingEvent.SCALE_UP
        assert decision.target_nodes > 2
    
    @pytest.mark.asyncio
    async def test_scale_down_evaluation(self, autoscaler, sample_nodes):
        """Test scale down decision evaluation."""
        # Add more nodes
        for i, node in enumerate(sample_nodes):
            autoscaler.add_node(node)
        
        # Add extra nodes
        for i in range(2, 4):
            extra_node = ComputeNode(
                node_id=f"node{i+1}",
                region="US-CA",
                available_cpus=8,
                available_memory_gb=32,
                available_gpus=1,
                gpu_type="V100",
                cost_per_hour=2.5,
                utilization=0.1,  # Very low utilization
                last_updated=datetime.now()
            )
            autoscaler.add_node(extra_node)
        
        decision = await autoscaler.evaluate_scaling_decision()
        
        assert decision is not None
        assert decision.event == ScalingEvent.SCALE_DOWN
        assert decision.target_nodes < 4
    
    @pytest.mark.asyncio
    async def test_carbon_mitigation_evaluation(self, autoscaler, sample_nodes):
        """Test carbon mitigation decision evaluation."""
        for node in sample_nodes:
            autoscaler.add_node(node)
        
        # Set very high carbon intensity
        now = datetime.now()
        autoscaler._carbon_history = [
            (now, 300.0, "US-CA"),  # Very high carbon
            (now, 250.0, "US-WA"),
            (now, 80.0, "EU-FR")    # Much cleaner region available
        ]
        
        decision = await autoscaler.evaluate_scaling_decision()
        
        assert decision is not None
        assert decision.event in [ScalingEvent.MIGRATE, ScalingEvent.PAUSE]
    
    @pytest.mark.asyncio
    async def test_cooldown_period(self, autoscaler, sample_nodes):
        """Test cooldown period enforcement."""
        for node in sample_nodes:
            autoscaler.add_node(node)
        
        # Set conditions that would normally trigger scaling
        for node in autoscaler.current_nodes.values():
            node.utilization = 0.9
        
        # Simulate recent scaling event
        autoscaler.last_scaling_event = datetime.now() - timedelta(seconds=30)
        
        decision = await autoscaler.evaluate_scaling_decision()
        
        # Should be None due to cooldown
        assert decision is None
    
    @pytest.mark.asyncio
    async def test_execute_scale_up(self, autoscaler):
        """Test executing scale up decision."""
        decision = ScalingDecision(
            event=ScalingEvent.SCALE_UP,
            target_nodes=3,
            reasoning="Test scale up"
        )
        
        success = await autoscaler.execute_scaling_decision(decision)
        
        assert success is True
        assert len(autoscaler.current_nodes) == 3
        assert len(autoscaler.scaling_history) == 1
    
    @pytest.mark.asyncio
    async def test_execute_scale_down(self, autoscaler, sample_nodes):
        """Test executing scale down decision."""
        # Add nodes first
        for node in sample_nodes:
            autoscaler.add_node(node)
        
        decision = ScalingDecision(
            event=ScalingEvent.SCALE_DOWN,
            target_nodes=1,
            reasoning="Test scale down"
        )
        
        success = await autoscaler.execute_scaling_decision(decision)
        
        assert success is True
        assert len(autoscaler.current_nodes) == 1
    
    @pytest.mark.asyncio
    async def test_execute_migration(self, autoscaler, sample_nodes):
        """Test executing migration decision."""
        for node in sample_nodes:
            autoscaler.add_node(node)
        
        decision = ScalingDecision(
            event=ScalingEvent.MIGRATE,
            target_nodes=2,
            target_region="EU-FR",
            reasoning="Test migration"
        )
        
        success = await autoscaler.execute_scaling_decision(decision)
        
        assert success is True
        # All nodes should now be in target region
        for node in autoscaler.current_nodes.values():
            assert node.region == "EU-FR"
    
    def test_scaling_callbacks(self, autoscaler):
        """Test scaling event callbacks."""
        callback_called = False
        received_decision = None
        
        def test_callback(decision):
            nonlocal callback_called, received_decision
            callback_called = True
            received_decision = decision
        
        autoscaler.add_scaling_callback(test_callback)
        
        # Execute a scaling decision
        asyncio.run(autoscaler.execute_scaling_decision(
            ScalingDecision(
                event=ScalingEvent.SCALE_UP,
                target_nodes=2,
                reasoning="Test callback"
            )
        ))
        
        assert callback_called is True
        assert received_decision is not None
        assert received_decision.event == ScalingEvent.SCALE_UP
    
    def test_cost_estimation(self, autoscaler, sample_nodes):
        """Test cost change estimation."""
        for node in sample_nodes:
            autoscaler.add_node(node)
        
        # Average cost is (2.5 + 2.0) / 2 = 2.25
        cost_change = autoscaler._estimate_cost_change(2, 4)  # Scale from 2 to 4 nodes
        expected = 2 * 2.25  # 2 additional nodes * average cost
        
        assert cost_change == expected
    
    def test_carbon_change_estimation(self, autoscaler):
        """Test carbon impact estimation."""
        # Set up carbon history
        now = datetime.now()
        autoscaler._carbon_history = [(now, 150.0, "US-CA")]
        
        carbon_change = autoscaler._estimate_carbon_change(2, 4)
        
        # Should be positive (more emissions) for scaling up
        assert carbon_change > 0
        
        # Should be negative for scaling down
        carbon_change_down = autoscaler._estimate_carbon_change(4, 2)
        assert carbon_change_down < 0
    
    def test_scaling_stats(self, autoscaler, sample_nodes):
        """Test scaling statistics generation."""
        for node in sample_nodes:
            autoscaler.add_node(node)
        
        # Add some scaling history
        decision = ScalingDecision(
            event=ScalingEvent.SCALE_UP,
            target_nodes=3,
            reasoning="Test"
        )
        autoscaler.scaling_history.append((datetime.now(), decision))
        
        stats = autoscaler.get_scaling_stats()
        
        assert stats["current_nodes"] == 2
        assert stats["min_nodes"] == 1
        assert stats["max_nodes"] == 5
        assert "average_utilization" in stats
        assert "scaling_decisions_24h" in stats
        assert "total_cost_per_hour" in stats


class TestResourceRequirement:
    """Test cases for ResourceRequirement dataclass."""
    
    def test_resource_requirement_creation(self):
        """Test creating resource requirements."""
        req = ResourceRequirement(
            cpu_cores=8,
            memory_gb=32,
            gpu_count=2,
            gpu_type="V100",
            storage_gb=200
        )
        
        assert req.cpu_cores == 8
        assert req.memory_gb == 32
        assert req.gpu_count == 2
        assert req.gpu_type == "V100"
        assert req.storage_gb == 200
        assert req.network_bandwidth_gbps == 1.0  # Default value


class TestComputeNode:
    """Test cases for ComputeNode dataclass."""
    
    def test_compute_node_creation(self):
        """Test creating compute nodes."""
        node = ComputeNode(
            node_id="test-node",
            region="US-CA",
            available_cpus=16,
            available_memory_gb=64,
            available_gpus=4,
            gpu_type="A100",
            cost_per_hour=5.0
        )
        
        assert node.node_id == "test-node"
        assert node.region == "US-CA"
        assert node.available_cpus == 16
        assert node.available_memory_gb == 64
        assert node.available_gpus == 4
        assert node.gpu_type == "A100"
        assert node.cost_per_hour == 5.0
        assert node.utilization == 0.0  # Default value


class TestScalingDecision:
    """Test cases for ScalingDecision dataclass."""
    
    def test_scaling_decision_creation(self):
        """Test creating scaling decisions."""
        decision = ScalingDecision(
            event=ScalingEvent.SCALE_UP,
            target_nodes=5,
            target_region="EU-FR",
            reasoning="High utilization detected",
            estimated_cost_change=10.0,
            estimated_carbon_change=-5.0,
            urgency=0.8
        )
        
        assert decision.event == ScalingEvent.SCALE_UP
        assert decision.target_nodes == 5
        assert decision.target_region == "EU-FR"
        assert decision.reasoning == "High utilization detected"
        assert decision.estimated_cost_change == 10.0
        assert decision.estimated_carbon_change == -5.0
        assert decision.urgency == 0.8