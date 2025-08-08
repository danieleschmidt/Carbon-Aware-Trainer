"""
Tests for federated carbon-aware learning framework.

This module tests the federated learning implementation including
privacy preservation, pattern sharing, and collaborative optimization.
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from carbon_aware_trainer.core.types import CarbonIntensity
from carbon_aware_trainer.research.federated_learning import (
    FederatedCarbonLearning,
    FederatedCarbonOrchestrator,
    FederatedParticipant,
    FederatedRole,
    PrivacyParameters,
    DifferentialPrivacy,
    SecureAggregation,
    CarbonPattern,
    PrivacyMechanism
)


@pytest.fixture
def sample_carbon_data():
    """Generate sample carbon intensity data for federated learning tests."""
    data = []
    base_time = datetime.now() - timedelta(hours=720)  # 30 days ago
    
    for i in range(720):  # 30 days of hourly data
        timestamp = base_time + timedelta(hours=i)
        
        # Create realistic regional patterns
        base_intensity = 100 + 40 * ((i % 24) / 24)  # Daily cycle
        base_intensity += 25 * ((i % 168) / 168)      # Weekly cycle  
        base_intensity += 15 * ((i % 720) / 720)      # Monthly trend
        base_intensity += (hash(str(i)) % 50) - 25    # Random variation
        
        renewable_pct = 0.25 + 0.5 * ((i % 24) / 24)  # Daily renewable pattern
        renewable_pct += 0.1 * ((i % 168) / 168)       # Weekly variation
        
        ci = CarbonIntensity(
            carbon_intensity=max(20, min(400, base_intensity)),
            timestamp=timestamp,
            region="TEST_REGION",
            renewable_percentage=max(0.0, min(1.0, renewable_pct))
        )
        data.append(ci)
    
    return data


@pytest.fixture
def privacy_params():
    """Create privacy parameters for testing."""
    return PrivacyParameters(
        epsilon=1.0,
        delta=1e-6,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        secure_aggregation=True
    )


@pytest.fixture
def test_participant():
    """Create test participant."""
    return FederatedParticipant(
        participant_id="test_participant_001",
        region="US-CA",
        organization="Test University",
        trust_score=0.9,
        data_quality_score=0.85,
        contribution_weight=1.0,
        privacy_level=1.0
    )


class TestDifferentialPrivacy:
    """Test cases for differential privacy mechanisms."""
    
    @pytest.fixture
    def dp_mechanism(self):
        """Create differential privacy mechanism."""
        return DifferentialPrivacy(epsilon=1.0, delta=1e-6)
    
    def test_initialization(self, dp_mechanism):
        """Test differential privacy initialization."""
        assert dp_mechanism.epsilon == 1.0
        assert dp_mechanism.delta == 1e-6
    
    def test_add_gaussian_noise(self, dp_mechanism):
        """Test Gaussian noise addition."""
        original_values = [100.0, 200.0, 150.0, 175.0, 125.0]
        noisy_values = dp_mechanism.add_gaussian_noise(original_values, sensitivity=1.0)
        
        assert len(noisy_values) == len(original_values)
        
        # Values should be different (with high probability)
        differences = [abs(orig - noisy) for orig, noisy in zip(original_values, noisy_values)]
        assert any(diff > 0.01 for diff in differences)  # At least some noise added
        
        # Values shouldn't be too different (noise should be reasonable)
        assert all(diff < 100 for diff in differences)  # Noise not excessive
    
    def test_add_laplace_noise(self, dp_mechanism):
        """Test Laplace noise addition."""
        original_values = [50.0, 75.0, 100.0]
        noisy_values = dp_mechanism.add_laplace_noise(original_values, sensitivity=1.0)
        
        assert len(noisy_values) == len(original_values)
        
        # Check noise characteristics
        differences = [abs(orig - noisy) for orig, noisy in zip(original_values, noisy_values)]
        assert any(diff > 0.01 for diff in differences)
    
    def test_clip_and_add_noise(self, dp_mechanism):
        """Test gradient clipping and noise addition."""
        gradients = {
            "layer1": [10.0, 20.0, 5.0],  # Large gradients that need clipping
            "layer2": [0.1, 0.2, 0.05]    # Small gradients
        }
        
        clipped_gradients = dp_mechanism.clip_and_add_noise(gradients, max_norm=1.0)
        
        assert len(clipped_gradients) == len(gradients)
        assert "layer1" in clipped_gradients
        assert "layer2" in clipped_gradients
        
        # Check that large gradients were clipped
        layer1_norm = sum(g**2 for g in clipped_gradients["layer1"])**0.5
        assert layer1_norm <= 1.1  # Allow for small noise addition


class TestSecureAggregation:
    """Test cases for secure aggregation protocol."""
    
    @pytest.fixture
    def secure_agg(self):
        """Create secure aggregation instance."""
        return SecureAggregation()
    
    def test_generate_secret_shares(self, secure_agg):
        """Test secret share generation."""
        participant_id = "participant_001"
        num_participants = 5
        
        shares = secure_agg.generate_secret_shares(participant_id, num_participants)
        
        assert isinstance(shares, dict)
        assert len(shares) == num_participants
        assert participant_id in secure_agg.participant_secrets
        
        # Shares should sum to the original secret (approximately, due to floating point)
        secret = secure_agg.participant_secrets[participant_id]
        shares_sum = sum(shares.values())
        assert abs(shares_sum - secret) < 1e-10
    
    def test_create_aggregation_mask(self, secure_agg):
        """Test aggregation mask creation."""
        participant_weights = {
            "layer1": [1.0, 2.0, 3.0],
            "layer2": [0.5, 1.5, 2.5]
        }
        
        masked_weights = secure_agg.create_aggregation_mask(participant_weights)
        
        assert len(masked_weights) == len(participant_weights)
        assert "layer1" in masked_weights
        assert "layer2" in masked_weights
        
        # Masked weights should be different from original
        for param_name in participant_weights.keys():
            original = participant_weights[param_name]
            masked = masked_weights[param_name]
            differences = [abs(o - m) for o, m in zip(original, masked)]
            assert any(diff > 0.01 for diff in differences)
    
    def test_remove_aggregation_mask(self, secure_agg):
        """Test aggregation mask removal."""
        participant_weights = {
            "layer1": [1.0, 2.0, 3.0],
            "layer2": [0.5, 1.5, 2.5]
        }
        
        # Create and then remove mask
        masked_weights = secure_agg.create_aggregation_mask(participant_weights)
        unmasked_weights = secure_agg.remove_aggregation_mask(masked_weights)
        
        # Should recover original weights (approximately)
        for param_name in participant_weights.keys():
            original = participant_weights[param_name]
            recovered = unmasked_weights[param_name]
            
            for o, r in zip(original, recovered):
                assert abs(o - r) < 1e-10


class TestFederatedCarbonLearning:
    """Test cases for federated carbon learning system."""
    
    @pytest.fixture
    def federated_participant(self, privacy_params):
        """Create federated learning participant."""
        return FederatedCarbonLearning(
            participant_id="test_participant",
            role=FederatedRole.PARTICIPANT,
            privacy_params=privacy_params
        )
    
    @pytest.fixture
    def federated_coordinator(self, privacy_params):
        """Create federated learning coordinator."""
        return FederatedCarbonLearning(
            participant_id="coordinator",
            role=FederatedRole.COORDINATOR,
            privacy_params=privacy_params
        )
    
    def test_initialization(self, federated_participant):
        """Test federated learning initialization."""
        assert federated_participant.participant_id == "test_participant"
        assert federated_participant.role == FederatedRole.PARTICIPANT
        assert federated_participant.privacy_params is not None
        assert federated_participant.round_number == 0
        assert federated_participant.privacy_budget_spent == 0.0
    
    @pytest.mark.asyncio
    async def test_register_participant(self, federated_coordinator, test_participant):
        """Test participant registration."""
        success = await federated_coordinator.register_participant(test_participant)
        
        assert success is True
        assert test_participant.participant_id in federated_coordinator.participants
        
        # Test duplicate registration
        success_duplicate = await federated_coordinator.register_participant(test_participant)
        assert success_duplicate is False
    
    def test_validate_participant(self, federated_coordinator):
        """Test participant validation."""
        # Valid participant
        valid_participant = FederatedParticipant(
            participant_id="valid_001",
            region="US-CA",
            organization="Valid Org",
            trust_score=0.8,
            data_quality_score=0.7
        )
        
        assert federated_coordinator._validate_participant(valid_participant) is True
        
        # Invalid participant (low trust score)
        invalid_participant = FederatedParticipant(
            participant_id="invalid_001",
            region="US-CA",
            organization="Invalid Org",
            trust_score=0.05,  # Below threshold
            data_quality_score=0.7
        )
        
        assert federated_coordinator._validate_participant(invalid_participant) is False
    
    @pytest.mark.asyncio
    async def test_extract_local_carbon_patterns(self, federated_participant, sample_carbon_data):
        """Test local carbon pattern extraction."""
        mock_forecaster = Mock()
        
        patterns = await federated_participant.extract_local_carbon_patterns(
            sample_carbon_data, mock_forecaster
        )
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        # Check pattern types
        pattern_types = set()
        for pattern in patterns:
            assert isinstance(pattern, CarbonPattern)
            assert pattern.region == "TEST_REGION"
            assert len(pattern.temporal_signature) > 0
            assert pattern.confidence_score >= 0
            pattern_types.add(pattern.pattern_id.split('_')[1])  # Extract pattern type
        
        # Should have multiple pattern types
        assert len(pattern_types) > 1
    
    @pytest.mark.asyncio
    async def test_extract_daily_pattern(self, federated_participant, sample_carbon_data):
        """Test daily pattern extraction."""
        pattern = await federated_participant._extract_daily_pattern(sample_carbon_data)
        
        assert pattern is not None
        assert isinstance(pattern, CarbonPattern)
        assert len(pattern.temporal_signature) == 24  # One value per hour
        assert pattern.region == "TEST_REGION"
        assert 0 <= pattern.confidence_score <= 1
        
        # Privacy noise should be applied
        assert pattern.privacy_noise_level == federated_participant.privacy_params.epsilon
    
    @pytest.mark.asyncio
    async def test_extract_weekly_pattern(self, federated_participant, sample_carbon_data):
        """Test weekly pattern extraction."""
        pattern = await federated_participant._extract_weekly_pattern(sample_carbon_data)
        
        assert pattern is not None
        assert isinstance(pattern, CarbonPattern)
        assert len(pattern.temporal_signature) == 7  # One value per day
        assert "weekly" in pattern.seasonal_components
    
    @pytest.mark.asyncio
    async def test_extract_renewable_pattern(self, federated_participant, sample_carbon_data):
        """Test renewable energy pattern extraction."""
        pattern = await federated_participant._extract_renewable_pattern(sample_carbon_data)
        
        assert pattern is not None
        assert isinstance(pattern, CarbonPattern)
        assert -1 <= pattern.renewable_correlation <= 1  # Valid correlation range
        assert len(pattern.temporal_signature) > 0
    
    @pytest.mark.asyncio
    async def test_share_patterns_privately(self, federated_participant, sample_carbon_data):
        """Test private pattern sharing."""
        mock_forecaster = Mock()
        patterns = await federated_participant.extract_local_carbon_patterns(
            sample_carbon_data[:100], mock_forecaster
        )
        
        shared_data = await federated_participant.share_patterns_privately(patterns)
        
        assert isinstance(shared_data, dict)
        assert "participant_id" in shared_data
        assert "timestamp" in shared_data
        assert "patterns" in shared_data
        assert "privacy_metadata" in shared_data
        
        # Check privacy metadata
        privacy_metadata = shared_data["privacy_metadata"]
        assert "epsilon" in privacy_metadata
        assert "noise_mechanism" in privacy_metadata
        
        # Privacy budget should be updated
        assert federated_participant.privacy_budget_spent > 0
    
    @pytest.mark.asyncio
    async def test_aggregate_federated_patterns(self, federated_coordinator):
        """Test federated pattern aggregation."""
        # Mock participant patterns
        participant_patterns = [
            {
                "participant_id": "participant_1",
                "patterns": [{
                    "seasonal_components": {"daily": [0.8, 0.9, 1.0, 1.1]},
                    "renewable_correlation": -0.3,
                    "confidence_score": 0.8,
                    "region": "US-CA"
                }]
            },
            {
                "participant_id": "participant_2", 
                "patterns": [{
                    "seasonal_components": {"daily": [0.9, 1.0, 1.1, 1.2]},
                    "renewable_correlation": -0.4,
                    "confidence_score": 0.7,
                    "region": "US-WA"
                }]
            }
        ]
        
        aggregated = await federated_coordinator.aggregate_federated_patterns(participant_patterns)
        
        assert isinstance(aggregated, dict)
        assert "aggregated_patterns" in aggregated
        assert "total_participants" in aggregated
        assert "round_number" in aggregated
        
        assert aggregated["total_participants"] == 2
        
        # Check aggregated daily pattern
        if "daily" in aggregated["aggregated_patterns"]:
            daily_pattern = aggregated["aggregated_patterns"]["daily"]
            assert "temporal_signature" in daily_pattern
            assert "renewable_correlation" in daily_pattern
            assert "participant_count" in daily_pattern
            assert daily_pattern["participant_count"] == 2
    
    @pytest.mark.asyncio
    async def test_apply_federated_insights(self, federated_participant):
        """Test application of federated insights to local forecaster."""
        # Mock federated patterns
        federated_patterns = {
            "aggregated_patterns": {
                "daily": {
                    "temporal_signature": [0.8, 0.9, 1.0, 1.1, 1.0, 0.9],
                    "confidence_score": 0.85,
                    "regions_represented": ["US-CA", "US-WA"]
                },
                "renewable": {
                    "renewable_correlation": -0.35,
                    "confidence_score": 0.75,
                    "regions_represented": ["EU-FR", "EU-NO"]
                }
            },
            "total_participants": 4
        }
        
        mock_forecaster = Mock()
        
        insights_applied = await federated_participant.apply_federated_insights(
            federated_patterns, mock_forecaster
        )
        
        assert isinstance(insights_applied, dict)
        assert "insights_applied" in insights_applied
        assert "improvement_estimates" in insights_applied
        assert "confidence_boost" in insights_applied
        
        # Should have applied multiple insights
        insights = insights_applied["insights_applied"]
        assert len(insights) > 0
        
        for insight in insights:
            assert "type" in insight
            assert "regions_learned_from" in insight
    
    @pytest.mark.asyncio
    async def test_evaluate_federated_performance(self, federated_participant):
        """Test federated performance evaluation."""
        baseline_metrics = {"mae": 15.0, "rmse": 20.0, "r2": 0.7}
        federated_metrics = {"mae": 12.0, "rmse": 18.0, "r2": 0.8}
        
        evaluation = await federated_participant.evaluate_federated_performance(
            baseline_metrics, federated_metrics
        )
        
        assert isinstance(evaluation, dict)
        assert "improvements" in evaluation
        assert "privacy_cost_benefit" in evaluation
        assert "overall_assessment" in evaluation
        
        # Check improvement calculations
        improvements = evaluation["improvements"]
        assert "mae" in improvements
        assert improvements["mae"]["relative_improvement_percent"] > 0  # MAE should improve
        
        # Check privacy cost-benefit analysis
        cost_benefit = evaluation["privacy_cost_benefit"]
        assert "privacy_budget_spent" in cost_benefit
        assert "improvement_per_epsilon" in cost_benefit
    
    def test_save_and_load_federated_state(self, federated_participant, test_participant):
        """Test saving and loading federated state."""
        # Setup some state
        federated_participant.round_number = 5
        federated_participant.privacy_budget_spent = 2.5
        federated_participant.participants[test_participant.participant_id] = test_participant
        
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
            # Save state
            federated_participant.save_federated_state(tmp_file.name)
            
            # Create new instance and load state
            new_participant = FederatedCarbonLearning(
                "temp_id", FederatedRole.PARTICIPANT
            )
            new_participant.load_federated_state(tmp_file.name)
            
            # Verify loaded state
            assert new_participant.participant_id == federated_participant.participant_id
            assert new_participant.role == federated_participant.role
            assert new_participant.round_number == 5
            assert new_participant.privacy_budget_spent == 2.5
            assert test_participant.participant_id in new_participant.participants
            
            import os
            os.unlink(tmp_file.name)  # Cleanup


class TestFederatedCarbonOrchestrator:
    """Test cases for federated carbon orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create federated orchestrator."""
        return FederatedCarbonOrchestrator("test_orchestrator")
    
    @pytest.fixture
    def mock_participants(self, privacy_params):
        """Create mock participants for orchestration."""
        participants = []
        
        # Coordinator
        coordinator = FederatedCarbonLearning(
            "coordinator", FederatedRole.COORDINATOR, privacy_params
        )
        participants.append(coordinator)
        
        # Participants
        for i in range(3):
            participant = FederatedCarbonLearning(
                f"participant_{i}", FederatedRole.PARTICIPANT, privacy_params
            )
            participants.append(participant)
        
        return participants
    
    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.orchestrator_id == "test_orchestrator"
        assert len(orchestrator.participants) == 0
        assert len(orchestrator.learning_rounds) == 0
    
    @pytest.mark.asyncio
    async def test_orchestrate_federated_learning(self, orchestrator, mock_participants):
        """Test federated learning orchestration."""
        with patch.object(
            mock_participants[0], 'aggregate_federated_patterns',
            return_value={"final_patterns": {"test": "pattern"}}
        ) as mock_aggregate:
            with patch.object(
                mock_participants[1], 'share_patterns_privately',
                return_value={"patterns": [{"test": "data"}]}
            ) as mock_share:
                with patch.object(
                    mock_participants[1], 'apply_federated_insights',
                    return_value={"insights": "applied"}
                ) as mock_apply:
                    
                    result = await orchestrator.orchestrate_federated_learning(
                        mock_participants, rounds=2, min_participants=3
                    )
        
        assert isinstance(result, dict)
        assert "orchestrator_id" in result
        assert "start_timestamp" in result
        assert "end_timestamp" in result
        assert "participants" in result
        assert "rounds_completed" in result
        assert "privacy_summary" in result
        
        # Should have participant IDs
        assert len(result["participants"]) == len(mock_participants)
        
        # Privacy summary should be calculated
        privacy_summary = result["privacy_summary"]
        assert "total_privacy_budget_spent" in privacy_summary
        assert "average_privacy_per_participant" in privacy_summary
    
    @pytest.mark.asyncio
    async def test_orchestrate_insufficient_participants(self, orchestrator):
        """Test orchestration with insufficient participants."""
        # Only provide 2 participants when minimum is 3
        participants = [
            FederatedCarbonLearning("p1", FederatedRole.PARTICIPANT),
            FederatedCarbonLearning("p2", FederatedRole.PARTICIPANT)
        ]
        
        with pytest.raises(ValueError, match="Need at least"):
            await orchestrator.orchestrate_federated_learning(
                participants, rounds=2, min_participants=3
            )


class TestIntegrationScenarios:
    """Integration test scenarios for federated learning."""
    
    @pytest.mark.asyncio
    async def test_complete_federated_workflow(self, sample_carbon_data, privacy_params):
        """Test complete federated learning workflow."""
        # Create participants from different regions
        participant1 = FederatedCarbonLearning(
            "university_ca", FederatedRole.PARTICIPANT, privacy_params
        )
        participant2 = FederatedCarbonLearning(
            "company_wa", FederatedRole.PARTICIPANT, privacy_params
        )
        coordinator = FederatedCarbonLearning(
            "research_center", FederatedRole.COORDINATOR, privacy_params
        )
        
        # Register participants
        test_participant1 = FederatedParticipant(
            "university_ca", "US-CA", "University of California",
            trust_score=0.9, data_quality_score=0.85
        )
        test_participant2 = FederatedParticipant(
            "company_wa", "US-WA", "Tech Company",
            trust_score=0.8, data_quality_score=0.9
        )
        
        await coordinator.register_participant(test_participant1)
        await coordinator.register_participant(test_participant2)
        
        # Extract patterns (using different subsets to simulate regional differences)
        mock_forecaster = Mock()
        patterns1 = await participant1.extract_local_carbon_patterns(
            sample_carbon_data[:300], mock_forecaster
        )
        patterns2 = await participant2.extract_local_carbon_patterns(
            sample_carbon_data[300:600], mock_forecaster
        )
        
        # Share patterns privately
        shared1 = await participant1.share_patterns_privately(patterns1)
        shared2 = await participant2.share_patterns_privately(patterns2)
        
        # Aggregate patterns at coordinator
        aggregated = await coordinator.aggregate_federated_patterns([shared1, shared2])
        
        # Apply insights back to participants
        insights1 = await participant1.apply_federated_insights(aggregated, mock_forecaster)
        insights2 = await participant2.apply_federated_insights(aggregated, mock_forecaster)
        
        # Verify complete workflow
        assert len(patterns1) > 0
        assert len(patterns2) > 0
        assert "aggregated_patterns" in aggregated
        assert aggregated["total_participants"] == 2
        assert insights1["confidence_boost"] > 0
        assert insights2["confidence_boost"] > 0
        
        # Verify privacy preservation
        assert participant1.privacy_budget_spent > 0
        assert participant2.privacy_budget_spent > 0
        assert participant1.privacy_budget_spent <= participant1.privacy_params.epsilon * 10  # Reasonable bound
    
    @pytest.mark.asyncio
    async def test_federated_learning_with_orchestrator(self, sample_carbon_data, privacy_params):
        """Test federated learning using orchestrator."""
        orchestrator = FederatedCarbonOrchestrator("integration_test")
        
        # Create participants
        coordinator = FederatedCarbonLearning(
            "coordinator", FederatedRole.COORDINATOR, privacy_params
        )
        participants = [
            coordinator,
            FederatedCarbonLearning("p1", FederatedRole.PARTICIPANT, privacy_params),
            FederatedCarbonLearning("p2", FederatedRole.PARTICIPANT, privacy_params),
            FederatedCarbonLearning("p3", FederatedRole.PARTICIPANT, privacy_params)
        ]
        
        # Mock the pattern extraction and sharing methods
        with patch.object(participants[1], 'share_patterns_privately', return_value={"patterns": []}):
            with patch.object(participants[2], 'share_patterns_privately', return_value={"patterns": []}):
                with patch.object(participants[3], 'share_patterns_privately', return_value={"patterns": []}):
                    with patch.object(coordinator, 'aggregate_federated_patterns', return_value={"aggregated_patterns": {}}):
                        with patch.object(participants[1], 'apply_federated_insights', return_value={"insights": "applied"}):
                            with patch.object(participants[2], 'apply_federated_insights', return_value={"insights": "applied"}):
                                with patch.object(participants[3], 'apply_federated_insights', return_value={"insights": "applied"}):
                                    
                                    result = await orchestrator.orchestrate_federated_learning(
                                        participants, rounds=2, min_participants=3
                                    )
        
        # Verify orchestration results
        assert result["rounds_completed"] > 0
        assert len(result["participants"]) == 4  # Including coordinator
        assert "privacy_summary" in result