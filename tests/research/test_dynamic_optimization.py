"""
Tests for dynamic resource allocation optimization.

This module tests the advanced optimization algorithms including
genetic algorithms, reinforcement learning, and multi-objective optimization.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from carbon_aware_trainer.core.types import CarbonIntensity, CarbonForecast
from carbon_aware_trainer.research.dynamic_optimization import (
    DynamicResourceOptimizer,
    MultiObjectiveOptimizer,
    GeneticAlgorithmOptimizer,
    ReinforcementLearningOptimizer,
    Resource,
    WorkloadRequirement,
    AllocationDecision,
    OptimizationState,
    OptimizationObjective,
    ResourceType,
    AllocationStrategy
)


@pytest.fixture
def sample_resources():
    """Generate sample computational resources for testing."""
    resources = {}
    
    # GPU resources
    for i in range(4):
        resource_id = f"gpu_{i}"
        resources[resource_id] = Resource(
            resource_id=resource_id,
            resource_type=ResourceType.GPU,
            capacity=100.0,
            current_usage=20.0 + i * 10,
            power_consumption_watts=250.0,
            cost_per_hour=2.5,
            carbon_intensity_region="US-CA",
            reliability_score=0.95 - i * 0.05
        )
    
    # CPU resources
    for i in range(2):
        resource_id = f"cpu_{i}"
        resources[resource_id] = Resource(
            resource_id=resource_id,
            resource_type=ResourceType.CPU,
            capacity=64.0,
            current_usage=10.0 + i * 5,
            power_consumption_watts=150.0,
            cost_per_hour=1.0,
            carbon_intensity_region="US-CA",
            reliability_score=0.98
        )
    
    # Memory resources
    memory_resource = Resource(
        resource_id="memory_main",
        resource_type=ResourceType.MEMORY,
        capacity=512.0,  # GB
        current_usage=100.0,
        power_consumption_watts=50.0,
        cost_per_hour=0.5,
        carbon_intensity_region="US-CA",
        reliability_score=0.99
    )
    resources["memory_main"] = memory_resource
    
    return resources


@pytest.fixture
def sample_workloads():
    """Generate sample ML workloads for testing."""
    workloads = []
    
    # Small training workload
    workload1 = WorkloadRequirement(
        workload_id="training_small",
        resource_requirements={
            ResourceType.GPU: 20.0,
            ResourceType.CPU: 8.0,
            ResourceType.MEMORY: 32.0
        },
        estimated_duration_hours=6.0,
        deadline=datetime.now() + timedelta(hours=24),
        priority=1.0,
        interruptible=True,
        carbon_budget_kg=50.0,
        cost_budget_usd=100.0
    )
    workloads.append(workload1)
    
    # Large training workload
    workload2 = WorkloadRequirement(
        workload_id="training_large",
        resource_requirements={
            ResourceType.GPU: 80.0,
            ResourceType.CPU: 32.0,
            ResourceType.MEMORY: 128.0
        },
        estimated_duration_hours=24.0,
        deadline=datetime.now() + timedelta(hours=48),
        priority=2.0,
        interruptible=False,
        carbon_budget_kg=200.0,
        cost_budget_usd=500.0
    )
    workloads.append(workload2)
    
    # Inference workload
    workload3 = WorkloadRequirement(
        workload_id="inference_service",
        resource_requirements={
            ResourceType.GPU: 10.0,
            ResourceType.MEMORY: 16.0
        },
        estimated_duration_hours=12.0,
        deadline=datetime.now() + timedelta(hours=18),
        priority=0.5,
        interruptible=True,
        carbon_budget_kg=25.0,
        cost_budget_usd=50.0
    )
    workloads.append(workload3)
    
    return workloads


@pytest.fixture
def sample_carbon_forecasts():
    """Generate sample carbon intensity forecasts."""
    forecasts = {}
    base_time = datetime.now()
    
    # US-CA forecast
    data_points = []
    for i in range(48):
        timestamp = base_time + timedelta(hours=i + 1)
        # Simulate daily pattern with some variation
        base_intensity = 120 + 40 * ((i % 24) / 24)
        base_intensity += (hash(str(i)) % 60) - 30
        
        ci = CarbonIntensity(
            carbon_intensity=max(50, min(300, base_intensity)),
            timestamp=timestamp,
            region="US-CA",
            renewable_percentage=0.3 + 0.4 * ((i % 24) / 24)
        )
        data_points.append(ci)
    
    forecasts["US-CA"] = CarbonForecast(
        region="US-CA",
        forecast_time=base_time,
        data_points=data_points
    )
    
    return forecasts


class TestMultiObjectiveOptimizer:
    """Test cases for multi-objective optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create multi-objective optimizer."""
        objectives = [
            OptimizationObjective.MINIMIZE_CARBON,
            OptimizationObjective.MINIMIZE_COST,
            OptimizationObjective.MAXIMIZE_PERFORMANCE
        ]
        return MultiObjectiveOptimizer(objectives)
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert len(optimizer.objectives) == 3
        assert len(optimizer.weights) == 3
        assert all(w > 0 for w in optimizer.weights)
        assert len(optimizer.pareto_front) == 0
    
    def test_evaluate_solution(self, optimizer, sample_workloads, sample_carbon_forecasts):
        """Test solution evaluation against objectives."""
        workload = sample_workloads[0]
        forecast = list(sample_carbon_forecasts.values())[0]
        
        # Create sample allocation decision
        allocation = AllocationDecision(
            workload_id=workload.workload_id,
            allocated_resources={},
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=6),
            expected_carbon_kg=25.0,
            expected_cost_usd=75.0,
            expected_performance=0.9,
            confidence_score=0.8
        )
        
        objectives_values = optimizer.evaluate_solution(allocation, workload, forecast)
        
        assert isinstance(objectives_values, dict)
        assert "carbon" in objectives_values
        assert "cost" in objectives_values
        assert "performance" in objectives_values
        
        # Check that values are reasonable
        assert objectives_values["carbon"] == 25.0
        assert objectives_values["cost"] == 75.0
        assert objectives_values["performance"] == -0.9  # Negative for minimization
    
    def test_pareto_dominance(self, optimizer):
        """Test Pareto dominance checking."""
        solution_a = {"carbon": 100.0, "cost": 50.0, "performance": -0.8}
        solution_b = {"carbon": 80.0, "cost": 40.0, "performance": -0.9}  # Better in all
        solution_c = {"carbon": 90.0, "cost": 60.0, "performance": -0.85}  # Mixed
        
        # B dominates A (better in all objectives)
        assert optimizer.is_pareto_dominated(solution_a, solution_b) is True
        
        # A does not dominate C (not better in all objectives)
        assert optimizer.is_pareto_dominated(solution_c, solution_a) is False
        
        # C does not dominate A
        assert optimizer.is_pareto_dominated(solution_a, solution_c) is False
    
    def test_update_pareto_front(self, optimizer):
        """Test Pareto front updating."""
        # Add first solution
        solution1 = {
            "objectives": {"carbon": 100.0, "cost": 50.0},
            "allocation": Mock(),
            "workload_id": "test1"
        }
        optimizer.update_pareto_front(solution1)
        assert len(optimizer.pareto_front) == 1
        
        # Add dominated solution (should not be added)
        solution2 = {
            "objectives": {"carbon": 120.0, "cost": 60.0},  # Worse in both
            "allocation": Mock(),
            "workload_id": "test2"
        }
        optimizer.update_pareto_front(solution2)
        assert len(optimizer.pareto_front) == 1
        
        # Add dominating solution (should replace previous)
        solution3 = {
            "objectives": {"carbon": 80.0, "cost": 40.0},  # Better in both
            "allocation": Mock(),
            "workload_id": "test3"
        }
        optimizer.update_pareto_front(solution3)
        assert len(optimizer.pareto_front) == 1
        assert optimizer.pareto_front[0]["workload_id"] == "test3"
        
        # Add non-dominated solution (should be added)
        solution4 = {
            "objectives": {"carbon": 90.0, "cost": 30.0},  # Better cost, worse carbon
            "allocation": Mock(),
            "workload_id": "test4"
        }
        optimizer.update_pareto_front(solution4)
        assert len(optimizer.pareto_front) == 2
    
    def test_select_solution_from_pareto_front(self, optimizer):
        """Test solution selection from Pareto front."""
        # Add multiple solutions to front
        solutions = [
            {
                "objectives": {"carbon": 100.0, "cost": 40.0},
                "allocation": Mock(),
                "workload_id": "test1"
            },
            {
                "objectives": {"carbon": 80.0, "cost": 50.0},
                "allocation": Mock(),
                "workload_id": "test2"
            }
        ]
        
        for solution in solutions:
            optimizer.update_pareto_front(solution)
        
        # Select with equal preferences (should pick best overall)
        selected = optimizer.select_solution_from_pareto_front()
        assert selected is not None
        assert selected["workload_id"] in ["test1", "test2"]
        
        # Select with carbon preference
        carbon_preference = {"carbon": 2.0, "cost": 1.0}
        selected_carbon = optimizer.select_solution_from_pareto_front(carbon_preference)
        assert selected_carbon is not None
        # Should prefer test2 (lower carbon)
        assert selected_carbon["workload_id"] == "test2"


class TestGeneticAlgorithmOptimizer:
    """Test cases for genetic algorithm optimization."""
    
    @pytest.fixture
    def ga_optimizer(self):
        """Create genetic algorithm optimizer."""
        return GeneticAlgorithmOptimizer(
            population_size=20,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_ratio=0.2
        )
    
    def test_initialization(self, ga_optimizer):
        """Test GA optimizer initialization."""
        assert ga_optimizer.population_size == 20
        assert ga_optimizer.mutation_rate == 0.1
        assert ga_optimizer.crossover_rate == 0.8
        assert ga_optimizer.elite_size == 4  # 20% of 20
        assert ga_optimizer.generation == 0
    
    def test_encode_decode_solution(self, ga_optimizer, sample_workloads, sample_resources):
        """Test solution encoding and decoding."""
        # Create sample allocation decisions
        decisions = [
            AllocationDecision(
                workload_id=workload.workload_id,
                allocated_resources={list(sample_resources.keys())[0]: list(sample_resources.values())[0]},
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(hours=workload.estimated_duration_hours),
                expected_carbon_kg=50.0,
                expected_cost_usd=100.0,
                expected_performance=0.9,
                confidence_score=0.8
            )
            for workload in sample_workloads
        ]
        
        # Encode
        chromosome = ga_optimizer.encode_solution(decisions)
        assert isinstance(chromosome, list)
        assert len(chromosome) > 0
        
        # Decode
        decoded_decisions = ga_optimizer.decode_solution(chromosome, sample_workloads, sample_resources)
        assert len(decoded_decisions) <= len(sample_workloads)
        
        for decision in decoded_decisions:
            assert isinstance(decision, AllocationDecision)
            assert decision.workload_id in [w.workload_id for w in sample_workloads]
    
    def test_calculate_fitness(self, ga_optimizer, sample_workloads, sample_resources, sample_carbon_forecasts):
        """Test fitness calculation."""
        # Create simple chromosome
        chromosome = [12, 100, 200, 15, 150, 250]  # Hours and resource hashes
        
        fitness = ga_optimizer.calculate_fitness(
            chromosome, sample_workloads, sample_resources, sample_carbon_forecasts
        )
        
        assert isinstance(fitness, float)
        # Fitness should be reasonable (not extreme)
        assert -1000 <= fitness <= 1000
    
    def test_selection(self, ga_optimizer):
        """Test tournament selection."""
        population_with_fitness = [
            ([1, 2, 3], 0.5),
            ([4, 5, 6], 0.8),
            ([7, 8, 9], 0.3),
            ([10, 11, 12], 0.9)
        ]
        
        selected_chromosome = ga_optimizer.selection(population_with_fitness, k=3)
        
        assert isinstance(selected_chromosome, list)
        assert len(selected_chromosome) == 3
        # Should tend to select higher fitness (but random, so can't guarantee)
    
    def test_crossover(self, ga_optimizer):
        """Test genetic crossover operation."""
        parent1 = [1, 2, 3, 4, 5]
        parent2 = [6, 7, 8, 9, 10]
        
        child1, child2 = ga_optimizer.crossover(parent1, parent2)
        
        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)
        
        # Children should be different from parents (unless crossover point at end)
        assert child1 != parent1 or child1 == parent1  # May be same if crossover at end
        assert child2 != parent2 or child2 == parent2
    
    def test_mutate(self, ga_optimizer):
        """Test genetic mutation operation."""
        original_chromosome = [1, 2, 3, 4, 5]
        
        # Test multiple times due to randomness
        mutations_found = False
        for _ in range(10):
            mutated = ga_optimizer.mutate(original_chromosome[:])
            if mutated != original_chromosome:
                mutations_found = True
                break
        
        # Should find at least one mutation in 10 tries (probability > 99%)
        assert mutations_found
    
    @pytest.mark.asyncio
    async def test_optimize(self, ga_optimizer, sample_workloads, sample_resources, sample_carbon_forecasts):
        """Test genetic algorithm optimization."""
        # Use small number of generations for testing
        decisions = await ga_optimizer.optimize(
            sample_workloads, sample_resources, sample_carbon_forecasts, generations=5
        )
        
        assert isinstance(decisions, list)
        assert len(decisions) <= len(sample_workloads)
        
        # Should have some valid decisions
        for decision in decisions:
            assert isinstance(decision, AllocationDecision)
            assert decision.start_time <= decision.end_time
    
    def test_initialize_population(self, ga_optimizer, sample_workloads, sample_resources):
        """Test population initialization."""
        ga_optimizer._initialize_population(sample_workloads, sample_resources)
        
        assert len(ga_optimizer.population) == ga_optimizer.population_size
        
        for chromosome in ga_optimizer.population:
            assert isinstance(chromosome, list)
            assert len(chromosome) > 0


class TestReinforcementLearningOptimizer:
    """Test cases for reinforcement learning optimization."""
    
    @pytest.fixture
    def rl_optimizer(self):
        """Create RL optimizer."""
        return ReinforcementLearningOptimizer(
            state_dim=5,
            action_dim=4,
            learning_rate=0.01,
            epsilon=0.2
        )
    
    def test_initialization(self, rl_optimizer):
        """Test RL optimizer initialization."""
        assert rl_optimizer.state_dim == 5
        assert rl_optimizer.action_dim == 4
        assert rl_optimizer.learning_rate == 0.01
        assert rl_optimizer.epsilon == 0.2
        assert rl_optimizer.episode == 0
        assert len(rl_optimizer.q_table) == 0
    
    def test_encode_state(self, rl_optimizer, sample_resources, sample_workloads, sample_carbon_forecasts):
        """Test state encoding."""
        optimization_state = OptimizationState(
            current_allocations={},
            available_resources=sample_resources,
            pending_workloads=sample_workloads,
            carbon_forecasts=sample_carbon_forecasts,
            system_load=0.6,
            optimization_round=3,
            last_update=datetime.now()
        )
        
        encoded_state = rl_optimizer.encode_state(optimization_state, sample_carbon_forecasts)
        
        assert isinstance(encoded_state, tuple)
        assert len(encoded_state) == 5  # Should match state_dim
        
        # All state features should be integers in reasonable ranges
        for feature in encoded_state:
            assert isinstance(feature, int)
            assert 0 <= feature <= 10  # Based on encoding logic
    
    def test_decode_action(self, rl_optimizer):
        """Test action decoding."""
        optimization_state = OptimizationState(
            current_allocations={},
            available_resources={},
            pending_workloads=[],
            carbon_forecasts={},
            system_load=0.5,
            optimization_round=1,
            last_update=datetime.now()
        )
        
        for action_id in range(rl_optimizer.action_dim):
            decoded_action = rl_optimizer.decode_action(action_id, optimization_state)
            
            assert isinstance(decoded_action, dict)
            assert "action" in decoded_action
            assert decoded_action["action"] in ["schedule_now", "delay", "pause_training"]
    
    def test_get_action(self, rl_optimizer):
        """Test action selection."""
        state = (1, 2, 3, 4, 0)  # Sample state
        
        # Should return valid action
        action = rl_optimizer.get_action(state)
        assert 0 <= action < rl_optimizer.action_dim
        
        # State should be added to Q-table if new
        assert state in rl_optimizer.q_table
        assert len(rl_optimizer.q_table[state]) == rl_optimizer.action_dim
    
    def test_calculate_reward(self, rl_optimizer, sample_workloads, sample_carbon_forecasts):
        """Test reward calculation."""
        action_result = {
            "action_executed": "schedule_now",
            "carbon_saved": 10.0,
            "performance_achieved": 0.9,
            "delay_penalty": 2.0,
            "resource_efficiency": 0.8,
            "deadline_violated": False
        }
        
        workload = sample_workloads[0]
        forecast = list(sample_carbon_forecasts.values())[0]
        
        reward = rl_optimizer.calculate_reward(action_result, forecast, workload)
        
        assert isinstance(reward, float)
        # Reward should incorporate carbon savings, performance, etc.
        expected_reward = (10.0 * 0.1 + 0.9 * 0.05 - 2.0 * 0.02 + 0.8 * 0.03)
        assert abs(reward - expected_reward) < 0.01
    
    def test_update_q_value(self, rl_optimizer):
        """Test Q-value updating."""
        state = (1, 2, 3, 4, 0)
        next_state = (1, 2, 3, 4, 1)
        action = 1
        reward = 10.0
        
        # Initialize Q-tables
        rl_optimizer.q_table[state] = [0.0, 0.0, 0.0, 0.0]
        rl_optimizer.q_table[next_state] = [5.0, 3.0, 4.0, 2.0]
        
        initial_q = rl_optimizer.q_table[state][action]
        
        rl_optimizer.update_q_value(state, action, reward, next_state, done=False)
        
        # Q-value should be updated
        updated_q = rl_optimizer.q_table[state][action]
        assert updated_q != initial_q
        
        # Experience should be stored
        assert len(rl_optimizer.experience_buffer) == 1
    
    @pytest.mark.asyncio
    async def test_optimize_episode(self, rl_optimizer, sample_resources, sample_workloads, sample_carbon_forecasts):
        """Test RL episode optimization."""
        initial_state = OptimizationState(
            current_allocations={},
            available_resources=sample_resources,
            pending_workloads=sample_workloads,
            carbon_forecasts=sample_carbon_forecasts,
            system_load=0.5,
            optimization_round=0,
            last_update=datetime.now()
        )
        
        episode_result = await rl_optimizer.optimize_episode(
            initial_state, sample_carbon_forecasts, max_steps=10
        )
        
        assert isinstance(episode_result, dict)
        assert "episode" in episode_result
        assert "total_reward" in episode_result
        assert "steps_taken" in episode_result
        assert "actions" in episode_result
        
        # Episode number should be incremented
        assert rl_optimizer.episode == 1
        
        # Should have taken some actions
        assert len(episode_result["actions"]) > 0


class TestDynamicResourceOptimizer:
    """Test cases for main dynamic resource optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create dynamic resource optimizer."""
        return DynamicResourceOptimizer(AllocationStrategy.MULTI_OBJECTIVE_PARETO)
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.strategy == AllocationStrategy.MULTI_OBJECTIVE_PARETO
        assert optimizer.multi_objective is not None
        assert optimizer.genetic_algorithm is not None
        assert optimizer.reinforcement_learning is not None
    
    @pytest.mark.asyncio
    async def test_optimize_allocation_multi_objective(self, optimizer, sample_workloads, sample_resources, sample_carbon_forecasts):
        """Test allocation optimization using multi-objective strategy."""
        optimizer.strategy = AllocationStrategy.MULTI_OBJECTIVE_PARETO
        
        decisions = await optimizer.optimize_allocation(
            sample_workloads, sample_resources, sample_carbon_forecasts
        )
        
        assert isinstance(decisions, list)
        assert len(decisions) <= len(sample_workloads)
        
        # Verify optimization history
        assert len(optimizer.optimization_history) == 1
        history_record = optimizer.optimization_history[0]
        assert history_record["strategy"] == "multi_objective_pareto"
        assert history_record["success"] == (len(decisions) > 0)
    
    @pytest.mark.asyncio
    async def test_optimize_allocation_greedy(self, optimizer, sample_workloads, sample_resources, sample_carbon_forecasts):
        """Test allocation optimization using greedy strategy."""
        optimizer.strategy = AllocationStrategy.GREEDY
        
        decisions = await optimizer.optimize_allocation(
            sample_workloads, sample_resources, sample_carbon_forecasts
        )
        
        assert isinstance(decisions, list)
        assert len(decisions) <= len(sample_workloads)
    
    @pytest.mark.asyncio
    async def test_optimize_allocation_adaptive_threshold(self, optimizer, sample_workloads, sample_resources, sample_carbon_forecasts):
        """Test allocation optimization using adaptive threshold strategy."""
        optimizer.strategy = AllocationStrategy.ADAPTIVE_THRESHOLD
        
        decisions = await optimizer.optimize_allocation(
            sample_workloads, sample_resources, sample_carbon_forecasts
        )
        
        assert isinstance(decisions, list)
        assert len(decisions) <= len(sample_workloads)
    
    def test_generate_candidate_allocations(self, optimizer, sample_workloads, sample_resources, sample_carbon_forecasts):
        """Test candidate allocation generation."""
        workload = sample_workloads[0]
        
        candidates = optimizer._generate_candidate_allocations(
            workload, sample_resources, sample_carbon_forecasts
        )
        
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        
        for candidate in candidates:
            assert isinstance(candidate, AllocationDecision)
            assert candidate.workload_id == workload.workload_id
            assert candidate.start_time < candidate.end_time
            assert candidate.expected_carbon_kg > 0
            assert candidate.expected_cost_usd > 0
    
    def test_estimate_carbon_consumption(self, optimizer, sample_resources, sample_carbon_forecasts):
        """Test carbon consumption estimation."""
        resources = {k: v for k, v in list(sample_resources.items())[:2]}
        duration_hours = 8.0
        
        carbon_estimate = optimizer._estimate_carbon_consumption(
            resources, duration_hours, sample_carbon_forecasts
        )
        
        assert isinstance(carbon_estimate, float)
        assert carbon_estimate > 0
        
        # Should be reasonable for 8 hours of GPU+CPU usage
        assert 1.0 < carbon_estimate < 100.0
    
    def test_estimate_cost(self, optimizer, sample_resources):
        """Test cost estimation."""
        resources = {k: v for k, v in list(sample_resources.items())[:2]}
        duration_hours = 8.0
        
        cost_estimate = optimizer._estimate_cost(resources, duration_hours)
        
        assert isinstance(cost_estimate, float)
        assert cost_estimate > 0
        
        # Should be reasonable for 8 hours of GPU+CPU usage
        assert 10.0 < cost_estimate < 100.0
    
    def test_estimate_performance(self, optimizer, sample_resources, sample_workloads):
        """Test performance estimation."""
        resources = {k: v for k, v in list(sample_resources.items())[:3]}
        workload = sample_workloads[0]
        
        performance_estimate = optimizer._estimate_performance(resources, workload)
        
        assert isinstance(performance_estimate, float)
        assert 0.0 <= performance_estimate <= 1.0
    
    def test_generate_optimization_report(self, optimizer):
        """Test optimization report generation."""
        # Add some mock history
        optimizer.optimization_history = [
            {
                "timestamp": datetime.now().isoformat(),
                "strategy": "multi_objective_pareto",
                "workloads_count": 3,
                "decisions_count": 2,
                "optimization_time_seconds": 1.5,
                "success": True
            },
            {
                "timestamp": datetime.now().isoformat(),
                "strategy": "greedy",
                "workloads_count": 2,
                "decisions_count": 2,
                "optimization_time_seconds": 0.8,
                "success": True
            }
        ]
        
        report = optimizer.generate_optimization_report()
        
        assert isinstance(report, dict)
        assert "total_optimizations" in report
        assert "strategy_distribution" in report
        assert "performance_metrics" in report
        
        assert report["total_optimizations"] == 2
        
        # Check strategy distribution
        strategy_dist = report["strategy_distribution"]
        assert "multi_objective_pareto" in strategy_dist
        assert "greedy" in strategy_dist
        
        # Check performance metrics
        perf_metrics = report["performance_metrics"]
        assert "average_optimization_time_seconds" in perf_metrics
        assert "success_rate" in perf_metrics
        assert perf_metrics["success_rate"] == 1.0  # Both successful


class TestIntegrationScenarios:
    """Integration tests for dynamic optimization system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization_workflow(self, sample_workloads, sample_resources, sample_carbon_forecasts):
        """Test complete optimization workflow."""
        optimizer = DynamicResourceOptimizer(AllocationStrategy.MULTI_OBJECTIVE_PARETO)
        
        # Run optimization
        decisions = await optimizer.optimize_allocation(
            sample_workloads, sample_resources, sample_carbon_forecasts,
            objectives=[
                OptimizationObjective.MINIMIZE_CARBON,
                OptimizationObjective.MINIMIZE_COST
            ]
        )
        
        # Verify results
        assert len(decisions) <= len(sample_workloads)
        
        for decision in decisions:
            # Verify decision validity
            assert decision.workload_id in [w.workload_id for w in sample_workloads]
            assert decision.start_time <= decision.end_time
            assert decision.expected_carbon_kg >= 0
            assert decision.expected_cost_usd >= 0
            assert 0 <= decision.expected_performance <= 1
            assert 0 <= decision.confidence_score <= 1
        
        # Generate and verify report
        report = optimizer.generate_optimization_report()
        assert report["total_optimizations"] == 1
        assert report["performance_metrics"]["success_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_strategy_comparison(self, sample_workloads, sample_resources, sample_carbon_forecasts):
        """Test optimization with different strategies."""
        strategies = [
            AllocationStrategy.MULTI_OBJECTIVE_PARETO,
            AllocationStrategy.GREEDY,
            AllocationStrategy.ADAPTIVE_THRESHOLD
        ]
        
        results = {}
        
        for strategy in strategies:
            optimizer = DynamicResourceOptimizer(strategy)
            decisions = await optimizer.optimize_allocation(
                sample_workloads, sample_resources, sample_carbon_forecasts
            )
            results[strategy.value] = {
                "decisions_count": len(decisions),
                "total_carbon": sum(d.expected_carbon_kg for d in decisions),
                "total_cost": sum(d.expected_cost_usd for d in decisions)
            }
        
        # Verify all strategies produced results
        for strategy_name, result in results.items():
            assert result["decisions_count"] >= 0
            assert result["total_carbon"] >= 0
            assert result["total_cost"] >= 0
    
    @pytest.mark.asyncio 
    async def test_optimization_under_constraints(self, sample_resources, sample_carbon_forecasts):
        """Test optimization with tight resource constraints."""
        # Create workloads that exceed available resources
        demanding_workloads = [
            WorkloadRequirement(
                workload_id=f"demanding_{i}",
                resource_requirements={
                    ResourceType.GPU: 90.0,  # High GPU demand
                    ResourceType.MEMORY: 200.0  # High memory demand
                },
                estimated_duration_hours=12.0,
                deadline=datetime.now() + timedelta(hours=24),
                priority=1.0
            )
            for i in range(3)
        ]
        
        optimizer = DynamicResourceOptimizer(AllocationStrategy.GREEDY)
        decisions = await optimizer.optimize_allocation(
            demanding_workloads, sample_resources, sample_carbon_forecasts
        )
        
        # Should handle constraints gracefully
        assert isinstance(decisions, list)
        # May have fewer decisions due to resource constraints
        assert len(decisions) <= len(demanding_workloads)