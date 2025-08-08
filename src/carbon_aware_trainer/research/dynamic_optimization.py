"""
Dynamic Resource Allocation Optimization for Carbon-Aware ML Training.

This module implements advanced algorithms for dynamic resource allocation that
adapts in real-time to carbon intensity variations, demand patterns, and
training requirements. Includes multi-objective optimization, reinforcement
learning, and game-theoretic approaches.
"""

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Callable
from enum import Enum
import json

# Optional numerical and optimization libraries
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from ..core.types import CarbonIntensity, CarbonForecast, TrainingMetrics
from ..core.monitor import CarbonMonitor
from ..core.advanced_forecasting import AdvancedCarbonForecaster


logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Multi-objective optimization objectives."""
    MINIMIZE_CARBON = "minimize_carbon"
    MINIMIZE_COST = "minimize_cost" 
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    MAXIMIZE_RENEWABLE = "maximize_renewable"
    MAXIMIZE_RELIABILITY = "maximize_reliability"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"


class AllocationStrategy(Enum):
    """Resource allocation strategies."""
    GREEDY = "greedy"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    MULTI_OBJECTIVE_PARETO = "multi_objective_pareto"
    GAME_THEORETIC = "game_theoretic"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"


@dataclass
class Resource:
    """Computational resource definition."""
    resource_id: str
    resource_type: ResourceType
    capacity: float
    current_usage: float
    power_consumption_watts: float
    cost_per_hour: float
    carbon_intensity_region: str
    availability_schedule: List[Tuple[datetime, datetime]] = field(default_factory=list)
    reliability_score: float = 1.0
    maintenance_windows: List[Tuple[datetime, datetime]] = field(default_factory=list)


@dataclass
class WorkloadRequirement:
    """ML training workload requirements."""
    workload_id: str
    resource_requirements: Dict[ResourceType, float]
    estimated_duration_hours: float
    deadline: Optional[datetime] = None
    priority: float = 1.0
    interruptible: bool = True
    checkpoint_frequency_hours: float = 1.0
    performance_target: float = 1.0
    carbon_budget_kg: Optional[float] = None
    cost_budget_usd: Optional[float] = None


@dataclass
class AllocationDecision:
    """Resource allocation decision."""
    workload_id: str
    allocated_resources: Dict[str, Resource]
    start_time: datetime
    end_time: datetime
    expected_carbon_kg: float
    expected_cost_usd: float
    expected_performance: float
    confidence_score: float
    migration_plan: Optional[List[Dict[str, Any]]] = None


@dataclass
class OptimizationState:
    """Current state for optimization algorithms."""
    current_allocations: Dict[str, AllocationDecision]
    available_resources: Dict[str, Resource]
    pending_workloads: List[WorkloadRequirement]
    carbon_forecasts: Dict[str, CarbonForecast]
    system_load: float
    optimization_round: int
    last_update: datetime


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for carbon-aware resource allocation.
    
    Implements Pareto-optimal solutions balancing carbon footprint,
    cost, performance, and reliability objectives.
    """
    
    def __init__(self, objectives: List[OptimizationObjective], weights: Optional[List[float]] = None):
        """Initialize multi-objective optimizer.
        
        Args:
            objectives: List of optimization objectives
            weights: Weights for objective combination (None for Pareto optimization)
        """
        self.objectives = objectives
        self.weights = weights or [1.0 / len(objectives)] * len(objectives)
        self.pareto_front = []
        self.solution_history = []
        
        if len(self.weights) != len(self.objectives):
            raise ValueError("Number of weights must match number of objectives")
        
        logger.info(f"Initialized multi-objective optimizer with {len(objectives)} objectives")
    
    def evaluate_solution(
        self,
        allocation: AllocationDecision,
        workload: WorkloadRequirement,
        carbon_forecast: CarbonForecast
    ) -> Dict[str, float]:
        """Evaluate a solution against all objectives."""
        objectives_values = {}
        
        # Carbon objective (minimize)
        if OptimizationObjective.MINIMIZE_CARBON in self.objectives:
            objectives_values['carbon'] = allocation.expected_carbon_kg
        
        # Cost objective (minimize)
        if OptimizationObjective.MINIMIZE_COST in self.objectives:
            objectives_values['cost'] = allocation.expected_cost_usd
        
        # Time objective (minimize completion time)
        if OptimizationObjective.MINIMIZE_TIME in self.objectives:
            completion_time = (allocation.end_time - allocation.start_time).total_seconds() / 3600
            objectives_values['time'] = completion_time
        
        # Performance objective (maximize)
        if OptimizationObjective.MAXIMIZE_PERFORMANCE in self.objectives:
            objectives_values['performance'] = -allocation.expected_performance  # Negative for minimization
        
        # Renewable energy objective (maximize renewable utilization)
        if OptimizationObjective.MAXIMIZE_RENEWABLE in self.objectives:
            # Calculate renewable percentage during allocation window
            renewable_score = self._calculate_renewable_score(allocation, carbon_forecast)
            objectives_values['renewable'] = -renewable_score  # Negative for minimization
        
        # Reliability objective (maximize)
        if OptimizationObjective.MAXIMIZE_RELIABILITY in self.objectives:
            reliability_score = allocation.confidence_score
            objectives_values['reliability'] = -reliability_score  # Negative for minimization
        
        return objectives_values
    
    def _calculate_renewable_score(self, allocation: AllocationDecision, carbon_forecast: CarbonForecast) -> float:
        """Calculate renewable energy utilization score."""
        if not carbon_forecast or not carbon_forecast.data_points:
            return 0.5  # Default neutral score
        
        # Find carbon intensities during allocation window
        allocation_intensities = []
        for ci in carbon_forecast.data_points:
            if allocation.start_time <= ci.timestamp <= allocation.end_time:
                if ci.renewable_percentage is not None:
                    allocation_intensities.append(ci.renewable_percentage)
        
        if not allocation_intensities:
            return 0.5
        
        return sum(allocation_intensities) / len(allocation_intensities)
    
    def is_pareto_dominated(self, solution_a: Dict[str, float], solution_b: Dict[str, float]) -> bool:
        """Check if solution A is dominated by solution B (B is better in all objectives)."""
        better_in_all = True
        better_in_at_least_one = False
        
        for obj_name in solution_a.keys():
            if solution_b[obj_name] < solution_a[obj_name]:  # B is better (lower value)
                better_in_at_least_one = True
            elif solution_b[obj_name] > solution_a[obj_name]:  # B is worse
                better_in_all = False
        
        return better_in_all and better_in_at_least_one
    
    def update_pareto_front(self, new_solution: Dict[str, Any]) -> None:
        """Update Pareto front with new solution."""
        new_objectives = new_solution['objectives']
        
        # Remove dominated solutions from current front
        non_dominated = []
        for existing_solution in self.pareto_front:
            if not self.is_pareto_dominated(existing_solution['objectives'], new_objectives):
                non_dominated.append(existing_solution)
        
        # Check if new solution is dominated by any existing solution
        dominated = False
        for existing_solution in self.pareto_front:
            if self.is_pareto_dominated(new_objectives, existing_solution['objectives']):
                dominated = True
                break
        
        if not dominated:
            non_dominated.append(new_solution)
            self.pareto_front = non_dominated
    
    def select_solution_from_pareto_front(self, preferences: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        """Select best solution from Pareto front based on preferences."""
        if not self.pareto_front:
            return None
        
        if preferences is None:
            preferences = {obj.value: 1.0 for obj in self.objectives}
        
        best_solution = None
        best_score = float('inf')
        
        for solution in self.pareto_front:
            # Calculate weighted score
            weighted_score = 0.0
            for obj_name, obj_value in solution['objectives'].items():
                weight = preferences.get(obj_name, 1.0)
                weighted_score += weight * obj_value
            
            if weighted_score < best_score:
                best_score = weighted_score
                best_solution = solution
        
        return best_solution


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for resource allocation optimization.
    
    Evolves population of allocation solutions to find optimal
    carbon-aware resource assignments.
    """
    
    def __init__(
        self,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1
    ):
        """Initialize genetic algorithm optimizer.
        
        Args:
            population_size: Size of evolution population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_ratio: Proportion of elite solutions to preserve
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = int(population_size * elite_ratio)
        
        self.population = []
        self.fitness_history = []
        self.generation = 0
        
        logger.info(f"Initialized GA optimizer: pop={population_size}, mut={mutation_rate}, cross={crossover_rate}")
    
    def encode_solution(self, allocation_decisions: List[AllocationDecision]) -> List[int]:
        """Encode allocation decisions as chromosome."""
        # Simple encoding: resource assignment indices
        chromosome = []
        
        for decision in allocation_decisions:
            # Encode start time as hour offset
            start_hour = decision.start_time.hour
            chromosome.append(start_hour)
            
            # Encode resource selection
            for resource_id in sorted(decision.allocated_resources.keys()):
                resource_hash = hash(resource_id) % 1000
                chromosome.append(resource_hash)
        
        return chromosome
    
    def decode_solution(
        self,
        chromosome: List[int],
        workloads: List[WorkloadRequirement],
        available_resources: Dict[str, Resource]
    ) -> List[AllocationDecision]:
        """Decode chromosome to allocation decisions."""
        decisions = []
        chromosome_idx = 0
        
        for workload in workloads:
            if chromosome_idx >= len(chromosome):
                break
            
            # Decode start time
            start_hour = chromosome[chromosome_idx] % 24
            start_time = datetime.now().replace(hour=start_hour, minute=0, second=0, microsecond=0)
            
            # Decode resource selection
            selected_resources = {}
            chromosome_idx += 1
            
            for resource_type, required_amount in workload.resource_requirements.items():
                if chromosome_idx < len(chromosome):
                    resource_hash = chromosome[chromosome_idx]
                    # Find resource with matching hash
                    for resource_id, resource in available_resources.items():
                        if resource.resource_type == resource_type and hash(resource_id) % 1000 == resource_hash:
                            selected_resources[resource_id] = resource
                            break
                    chromosome_idx += 1
            
            # Create allocation decision
            end_time = start_time + timedelta(hours=workload.estimated_duration_hours)
            
            decision = AllocationDecision(
                workload_id=workload.workload_id,
                allocated_resources=selected_resources,
                start_time=start_time,
                end_time=end_time,
                expected_carbon_kg=100.0,  # Would calculate based on resources
                expected_cost_usd=50.0,    # Would calculate based on resources
                expected_performance=1.0,   # Would estimate from resource allocation
                confidence_score=0.8
            )
            
            decisions.append(decision)
        
        return decisions
    
    def calculate_fitness(
        self,
        chromosome: List[int],
        workloads: List[WorkloadRequirement],
        available_resources: Dict[str, Resource],
        carbon_forecasts: Dict[str, CarbonForecast]
    ) -> float:
        """Calculate fitness score for chromosome."""
        try:
            decisions = self.decode_solution(chromosome, workloads, available_resources)
            
            total_fitness = 0.0
            
            for decision in decisions:
                # Carbon penalty (lower is better)
                carbon_penalty = decision.expected_carbon_kg * 0.01
                
                # Cost penalty
                cost_penalty = decision.expected_cost_usd * 0.001
                
                # Performance bonus
                performance_bonus = decision.expected_performance * 0.1
                
                # Reliability bonus
                reliability_bonus = decision.confidence_score * 0.05
                
                # Resource utilization efficiency
                utilization_bonus = self._calculate_utilization_bonus(decision)
                
                fitness_score = (
                    performance_bonus + reliability_bonus + utilization_bonus -
                    carbon_penalty - cost_penalty
                )
                
                total_fitness += fitness_score
            
            return total_fitness
            
        except Exception as e:
            logger.warning(f"Fitness calculation failed: {e}")
            return -1000.0  # Penalty for invalid solutions
    
    def _calculate_utilization_bonus(self, decision: AllocationDecision) -> float:
        """Calculate resource utilization efficiency bonus."""
        if not decision.allocated_resources:
            return 0.0
        
        utilization_scores = []
        for resource in decision.allocated_resources.values():
            # Assume optimal utilization is around 70-80%
            optimal_utilization = 0.75
            actual_utilization = resource.current_usage / resource.capacity if resource.capacity > 0 else 0
            
            # Bonus for near-optimal utilization
            utilization_diff = abs(actual_utilization - optimal_utilization)
            utilization_score = max(0, 0.1 - utilization_diff * 0.2)
            utilization_scores.append(utilization_score)
        
        return sum(utilization_scores) / len(utilization_scores)
    
    def selection(self, population: List[Tuple[List[int], float]], k: int = 3) -> List[int]:
        """Tournament selection for parent selection."""
        # Select k random individuals and return the best
        tournament = random.sample(population, min(k, len(population)))
        tournament.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness (higher is better)
        return tournament[0][0]  # Return chromosome
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Single-point crossover."""
        if len(parent1) != len(parent2) or len(parent1) < 2:
            return parent1[:], parent2[:]
        
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutate(self, chromosome: List[int]) -> List[int]:
        """Random mutation."""
        mutated = chromosome[:]
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Random mutation within reasonable bounds
                if i == 0:  # Start time gene
                    mutated[i] = random.randint(0, 23)
                else:  # Resource selection gene
                    mutated[i] = random.randint(0, 999)
        
        return mutated
    
    async def optimize(
        self,
        workloads: List[WorkloadRequirement],
        available_resources: Dict[str, Resource],
        carbon_forecasts: Dict[str, CarbonForecast],
        generations: int = 100
    ) -> List[AllocationDecision]:
        """Run genetic algorithm optimization."""
        logger.info(f"Starting GA optimization for {len(workloads)} workloads, {generations} generations")
        
        # Initialize population
        if not self.population:
            self._initialize_population(workloads, available_resources)
        
        best_fitness = float('-inf')
        best_solution = None
        
        for generation in range(generations):
            # Evaluate fitness for entire population
            population_with_fitness = []
            for chromosome in self.population:
                fitness = self.calculate_fitness(chromosome, workloads, available_resources, carbon_forecasts)
                population_with_fitness.append((chromosome, fitness))
            
            # Sort by fitness (descending)
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)
            
            # Track best solution
            current_best_fitness = population_with_fitness[0][1]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = population_with_fitness[0][0]
            
            # Elite selection
            new_population = [individual[0] for individual in population_with_fitness[:self.elite_size]]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.selection(population_with_fitness)
                parent2 = self.selection(population_with_fitness)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            self.population = new_population[:self.population_size]
            self.generation = generation
            
            # Track fitness history
            avg_fitness = sum(f for _, f in population_with_fitness) / len(population_with_fitness)
            self.fitness_history.append({
                'generation': generation,
                'best_fitness': current_best_fitness,
                'avg_fitness': avg_fitness
            })
            
            if generation % 20 == 0:
                logger.info(f"GA Generation {generation}: best={current_best_fitness:.3f}, avg={avg_fitness:.3f}")
        
        # Decode best solution
        if best_solution:
            final_decisions = self.decode_solution(best_solution, workloads, available_resources)
            logger.info(f"GA optimization completed: best fitness = {best_fitness:.3f}")
            return final_decisions
        else:
            logger.warning("GA optimization failed to find valid solution")
            return []
    
    def _initialize_population(self, workloads: List[WorkloadRequirement], available_resources: Dict[str, Resource]) -> None:
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.population_size):
            chromosome = []
            
            for workload in workloads:
                # Random start time
                start_hour = random.randint(0, 23)
                chromosome.append(start_hour)
                
                # Random resource selection
                for resource_type in workload.resource_requirements.keys():
                    # Find resources of this type
                    matching_resources = [
                        r_id for r_id, r in available_resources.items() 
                        if r.resource_type == resource_type
                    ]
                    
                    if matching_resources:
                        selected_resource = random.choice(matching_resources)
                        resource_hash = hash(selected_resource) % 1000
                        chromosome.append(resource_hash)
                    else:
                        chromosome.append(random.randint(0, 999))
            
            self.population.append(chromosome)


class ReinforcementLearningOptimizer:
    """
    Reinforcement Learning agent for dynamic resource allocation.
    
    Learns optimal allocation policies through interaction with
    the carbon-aware training environment.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        epsilon: float = 0.1,
        gamma: float = 0.95
    ):
        """Initialize RL optimizer.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for Q-learning
            epsilon: Exploration rate
            gamma: Discount factor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        
        # Q-table (simplified - would use neural network for large state spaces)
        self.q_table = {}
        self.episode = 0
        self.total_reward = 0.0
        
        # Experience replay
        self.experience_buffer = []
        self.buffer_size = 10000
        
        logger.info(f"Initialized RL optimizer: state_dim={state_dim}, action_dim={action_dim}")
    
    def encode_state(
        self,
        optimization_state: OptimizationState,
        carbon_forecasts: Dict[str, CarbonForecast]
    ) -> Tuple[int, ...]:
        """Encode current state for RL agent."""
        state_features = []
        
        # System load
        load_bucket = int(optimization_state.system_load * 10)  # 0-10
        state_features.append(min(10, max(0, load_bucket)))
        
        # Number of pending workloads
        pending_count = min(10, len(optimization_state.pending_workloads))
        state_features.append(pending_count)
        
        # Resource utilization
        if optimization_state.available_resources:
            total_capacity = sum(r.capacity for r in optimization_state.available_resources.values())
            total_usage = sum(r.current_usage for r in optimization_state.available_resources.values())
            utilization = int((total_usage / total_capacity) * 10) if total_capacity > 0 else 0
            state_features.append(min(10, max(0, utilization)))
        else:
            state_features.append(5)  # Default neutral state
        
        # Carbon intensity level
        if carbon_forecasts:
            avg_carbon = 0.0
            count = 0
            for forecast in carbon_forecasts.values():
                if forecast.data_points:
                    current_ci = forecast.data_points[0].carbon_intensity
                    avg_carbon += current_ci
                    count += 1
            
            if count > 0:
                avg_carbon /= count
                carbon_level = int(min(10, max(0, (avg_carbon - 50) / 30)))  # Normalize 50-350 to 0-10
                state_features.append(carbon_level)
            else:
                state_features.append(5)  # Default neutral
        else:
            state_features.append(5)
        
        # Time of day (0-23 mapped to 0-3 buckets)
        hour = datetime.now().hour
        time_bucket = hour // 6  # 0: 0-5, 1: 6-11, 2: 12-17, 3: 18-23
        state_features.append(time_bucket)
        
        return tuple(state_features)
    
    def decode_action(self, action: int, optimization_state: OptimizationState) -> Dict[str, Any]:
        """Decode action to allocation decision."""
        # Simplified action space: schedule_now, delay_1h, delay_2h, pause_training
        action_map = {
            0: {"action": "schedule_now", "delay_hours": 0},
            1: {"action": "delay", "delay_hours": 1},
            2: {"action": "delay", "delay_hours": 2},
            3: {"action": "pause_training", "delay_hours": 0}
        }
        
        return action_map.get(action, action_map[0])
    
    def get_action(self, state: Tuple[int, ...]) -> int:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best known action
            if state in self.q_table:
                q_values = self.q_table[state]
                return max(range(len(q_values)), key=lambda i: q_values[i])
            else:
                # Initialize Q-values for new state
                self.q_table[state] = [0.0] * self.action_dim
                return random.randint(0, self.action_dim - 1)
    
    def calculate_reward(
        self,
        action_result: Dict[str, Any],
        carbon_forecast: CarbonForecast,
        workload: WorkloadRequirement
    ) -> float:
        """Calculate reward for action."""
        reward = 0.0
        
        # Carbon efficiency reward
        if "carbon_saved" in action_result:
            carbon_reward = action_result["carbon_saved"] * 0.1  # Reward carbon savings
            reward += carbon_reward
        
        # Performance reward
        if "performance_achieved" in action_result:
            performance_reward = action_result["performance_achieved"] * 0.05
            reward += performance_reward
        
        # Timeliness penalty
        if "delay_penalty" in action_result:
            delay_penalty = action_result["delay_penalty"] * -0.02
            reward += delay_penalty
        
        # Resource efficiency reward
        if "resource_efficiency" in action_result:
            efficiency_reward = action_result["resource_efficiency"] * 0.03
            reward += efficiency_reward
        
        # Deadline violation penalty
        if action_result.get("deadline_violated", False):
            reward -= 10.0  # Large penalty for missing deadlines
        
        return reward
    
    def update_q_value(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool
    ) -> None:
        """Update Q-value using Q-learning."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.action_dim
        
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * self.action_dim
        
        # Q-learning update
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            max_next_q = max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[state][action] = current_q + self.learning_rate * (target - current_q)
        
        # Store experience
        experience = (state, action, reward, next_state, done)
        if len(self.experience_buffer) >= self.buffer_size:
            self.experience_buffer.pop(0)  # Remove oldest experience
        self.experience_buffer.append(experience)
    
    async def optimize_episode(
        self,
        initial_state: OptimizationState,
        carbon_forecasts: Dict[str, CarbonForecast],
        max_steps: int = 100
    ) -> Dict[str, Any]:
        """Run one optimization episode."""
        logger.info(f"Starting RL optimization episode {self.episode + 1}")
        
        current_state = initial_state
        episode_reward = 0.0
        actions_taken = []
        
        for step in range(max_steps):
            # Encode state
            state_encoded = self.encode_state(current_state, carbon_forecasts)
            
            # Select action
            action = self.get_action(state_encoded)
            action_decoded = self.decode_action(action, current_state)
            
            # Execute action (simulate environment response)
            action_result = await self._simulate_action_execution(action_decoded, current_state, carbon_forecasts)
            
            # Calculate reward
            reward = self.calculate_reward(action_result, list(carbon_forecasts.values())[0], current_state.pending_workloads[0] if current_state.pending_workloads else None)
            episode_reward += reward
            
            # Update state
            next_state = self._update_state(current_state, action_result)
            next_state_encoded = self.encode_state(next_state, carbon_forecasts)
            
            # Check if episode is done
            done = len(next_state.pending_workloads) == 0 or step == max_steps - 1
            
            # Update Q-value
            self.update_q_value(state_encoded, action, reward, next_state_encoded, done)
            
            actions_taken.append({
                "step": step,
                "action": action_decoded,
                "reward": reward,
                "state": state_encoded
            })
            
            # Move to next state
            current_state = next_state
            
            if done:
                break
        
        self.episode += 1
        self.total_reward += episode_reward
        
        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
        episode_result = {
            "episode": self.episode,
            "total_reward": episode_reward,
            "steps_taken": len(actions_taken),
            "actions": actions_taken,
            "final_state": current_state,
            "epsilon": self.epsilon
        }
        
        logger.info(f"RL episode completed: reward={episode_reward:.2f}, steps={len(actions_taken)}, ε={self.epsilon:.3f}")
        
        return episode_result
    
    async def _simulate_action_execution(
        self,
        action: Dict[str, Any],
        state: OptimizationState,
        carbon_forecasts: Dict[str, CarbonForecast]
    ) -> Dict[str, Any]:
        """Simulate execution of action in environment."""
        # Simplified simulation
        result = {
            "action_executed": action["action"],
            "carbon_saved": 0.0,
            "performance_achieved": 1.0,
            "delay_penalty": 0.0,
            "resource_efficiency": 0.8,
            "deadline_violated": False
        }
        
        if action["action"] == "schedule_now":
            # Immediate scheduling
            current_carbon = 100.0  # Would get from forecast
            if current_carbon < 80:  # Low carbon
                result["carbon_saved"] = 10.0
                result["performance_achieved"] = 1.0
            else:  # High carbon
                result["carbon_saved"] = -5.0
                result["performance_achieved"] = 0.9
        
        elif action["action"] == "delay":
            # Delayed scheduling
            delay_hours = action["delay_hours"]
            result["delay_penalty"] = delay_hours * 2.0
            
            # Assume carbon will be better after delay (simplified)
            result["carbon_saved"] = delay_hours * 3.0
            result["performance_achieved"] = 0.95
        
        elif action["action"] == "pause_training":
            # Pause current training
            result["carbon_saved"] = 15.0  # Large carbon savings
            result["performance_achieved"] = 0.7  # Performance impact
            result["delay_penalty"] = 5.0
        
        return result
    
    def _update_state(self, current_state: OptimizationState, action_result: Dict[str, Any]) -> OptimizationState:
        """Update state based on action execution."""
        new_state = OptimizationState(
            current_allocations=current_state.current_allocations.copy(),
            available_resources=current_state.available_resources.copy(),
            pending_workloads=current_state.pending_workloads.copy(),
            carbon_forecasts=current_state.carbon_forecasts,
            system_load=current_state.system_load,
            optimization_round=current_state.optimization_round + 1,
            last_update=datetime.now()
        )
        
        # Simulate state changes based on action
        if action_result["action_executed"] == "schedule_now" and new_state.pending_workloads:
            # Remove scheduled workload
            new_state.pending_workloads.pop(0)
            new_state.system_load = min(1.0, new_state.system_load + 0.1)
        
        elif action_result["action_executed"] == "pause_training":
            new_state.system_load = max(0.0, new_state.system_load - 0.2)
        
        return new_state


class DynamicResourceOptimizer:
    """
    Main dynamic resource optimization system combining multiple algorithms.
    
    Orchestrates different optimization strategies and adapts to changing
    conditions in real-time for carbon-aware ML training.
    """
    
    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.MULTI_OBJECTIVE_PARETO):
        """Initialize dynamic resource optimizer.
        
        Args:
            strategy: Primary optimization strategy to use
        """
        self.strategy = strategy
        
        # Initialize optimizers
        self.multi_objective = MultiObjectiveOptimizer([
            OptimizationObjective.MINIMIZE_CARBON,
            OptimizationObjective.MINIMIZE_COST,
            OptimizationObjective.MAXIMIZE_PERFORMANCE
        ])
        
        self.genetic_algorithm = GeneticAlgorithmOptimizer()
        self.reinforcement_learning = ReinforcementLearningOptimizer(state_dim=5, action_dim=4)
        
        # System state
        self.current_state = None
        self.optimization_history = []
        self.performance_metrics = {}
        
        logger.info(f"Initialized dynamic resource optimizer with strategy: {strategy.value}")
    
    async def optimize_allocation(
        self,
        workloads: List[WorkloadRequirement],
        available_resources: Dict[str, Resource],
        carbon_forecasts: Dict[str, CarbonForecast],
        objectives: Optional[List[OptimizationObjective]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[AllocationDecision]:
        """Optimize resource allocation using selected strategy."""
        logger.info(f"Optimizing allocation for {len(workloads)} workloads using {self.strategy.value}")
        
        start_time = datetime.now()
        
        # Update current state
        self.current_state = OptimizationState(
            current_allocations={},
            available_resources=available_resources,
            pending_workloads=workloads,
            carbon_forecasts=carbon_forecasts,
            system_load=0.5,  # Would calculate actual load
            optimization_round=len(self.optimization_history),
            last_update=start_time
        )
        
        allocation_decisions = []
        
        try:
            if self.strategy == AllocationStrategy.MULTI_OBJECTIVE_PARETO:
                allocation_decisions = await self._optimize_multi_objective(workloads, available_resources, carbon_forecasts, objectives)
            
            elif self.strategy == AllocationStrategy.GENETIC_ALGORITHM:
                allocation_decisions = await self.genetic_algorithm.optimize(workloads, available_resources, carbon_forecasts)
            
            elif self.strategy == AllocationStrategy.REINFORCEMENT_LEARNING:
                rl_result = await self.reinforcement_learning.optimize_episode(self.current_state, carbon_forecasts)
                allocation_decisions = self._convert_rl_result_to_allocations(rl_result, workloads, available_resources)
            
            elif self.strategy == AllocationStrategy.GREEDY:
                allocation_decisions = await self._optimize_greedy(workloads, available_resources, carbon_forecasts)
            
            elif self.strategy == AllocationStrategy.ADAPTIVE_THRESHOLD:
                allocation_decisions = await self._optimize_adaptive_threshold(workloads, available_resources, carbon_forecasts)
            
            else:
                logger.warning(f"Unknown strategy {self.strategy}, falling back to greedy")
                allocation_decisions = await self._optimize_greedy(workloads, available_resources, carbon_forecasts)
        
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            allocation_decisions = []
        
        # Record optimization results
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        optimization_record = {
            "timestamp": start_time.isoformat(),
            "strategy": self.strategy.value,
            "workloads_count": len(workloads),
            "decisions_count": len(allocation_decisions),
            "optimization_time_seconds": optimization_time,
            "success": len(allocation_decisions) > 0
        }
        
        self.optimization_history.append(optimization_record)
        
        logger.info(f"Optimization completed: {len(allocation_decisions)} decisions in {optimization_time:.3f}s")
        
        return allocation_decisions
    
    async def _optimize_multi_objective(
        self,
        workloads: List[WorkloadRequirement],
        available_resources: Dict[str, Resource],
        carbon_forecasts: Dict[str, CarbonForecast],
        objectives: Optional[List[OptimizationObjective]]
    ) -> List[AllocationDecision]:
        """Optimize using multi-objective Pareto approach."""
        if objectives:
            self.multi_objective.objectives = objectives
        
        decisions = []
        
        for workload in workloads:
            # Generate candidate allocations
            candidates = self._generate_candidate_allocations(workload, available_resources, carbon_forecasts)
            
            # Evaluate each candidate against objectives
            for candidate in candidates:
                objectives_values = self.multi_objective.evaluate_solution(
                    candidate, workload, list(carbon_forecasts.values())[0] if carbon_forecasts else None
                )
                
                candidate_solution = {
                    "allocation": candidate,
                    "objectives": objectives_values,
                    "workload_id": workload.workload_id
                }
                
                self.multi_objective.update_pareto_front(candidate_solution)
            
            # Select best solution for this workload
            best_solution = self.multi_objective.select_solution_from_pareto_front()
            if best_solution:
                decisions.append(best_solution["allocation"])
        
        return decisions
    
    def _generate_candidate_allocations(
        self,
        workload: WorkloadRequirement,
        available_resources: Dict[str, Resource],
        carbon_forecasts: Dict[str, CarbonForecast]
    ) -> List[AllocationDecision]:
        """Generate candidate resource allocations for workload."""
        candidates = []
        
        # Time windows to consider
        start_times = [
            datetime.now() + timedelta(hours=h) for h in range(0, 24, 4)
        ]
        
        for start_time in start_times:
            # Find suitable resources
            suitable_resources = {}
            
            for resource_type, required_amount in workload.resource_requirements.items():
                # Find resources of this type with sufficient capacity
                for resource_id, resource in available_resources.items():
                    if (resource.resource_type == resource_type and 
                        resource.capacity - resource.current_usage >= required_amount):
                        suitable_resources[resource_id] = resource
                        break  # Take first suitable resource (could be more sophisticated)
            
            if len(suitable_resources) >= len(workload.resource_requirements):
                end_time = start_time + timedelta(hours=workload.estimated_duration_hours)
                
                # Estimate metrics
                expected_carbon = self._estimate_carbon_consumption(suitable_resources, workload.estimated_duration_hours, carbon_forecasts)
                expected_cost = self._estimate_cost(suitable_resources, workload.estimated_duration_hours)
                expected_performance = self._estimate_performance(suitable_resources, workload)
                
                candidate = AllocationDecision(
                    workload_id=workload.workload_id,
                    allocated_resources=suitable_resources,
                    start_time=start_time,
                    end_time=end_time,
                    expected_carbon_kg=expected_carbon,
                    expected_cost_usd=expected_cost,
                    expected_performance=expected_performance,
                    confidence_score=0.8
                )
                
                candidates.append(candidate)
        
        return candidates
    
    def _estimate_carbon_consumption(
        self,
        resources: Dict[str, Resource],
        duration_hours: float,
        carbon_forecasts: Dict[str, CarbonForecast]
    ) -> float:
        """Estimate carbon consumption for resource allocation."""
        total_power_watts = sum(r.power_consumption_watts for r in resources.values())
        total_energy_kwh = (total_power_watts * duration_hours) / 1000.0
        
        # Get average carbon intensity
        avg_carbon_intensity = 100.0  # Default fallback
        
        if carbon_forecasts:
            total_intensity = 0.0
            count = 0
            for forecast in carbon_forecasts.values():
                if forecast.data_points:
                    for ci in forecast.data_points[:int(duration_hours)]:
                        total_intensity += ci.carbon_intensity
                        count += 1
            
            if count > 0:
                avg_carbon_intensity = total_intensity / count
        
        # Carbon emission in kg CO2
        carbon_emission_kg = (total_energy_kwh * avg_carbon_intensity) / 1000.0
        
        return carbon_emission_kg
    
    def _estimate_cost(self, resources: Dict[str, Resource], duration_hours: float) -> float:
        """Estimate cost for resource allocation."""
        total_cost = sum(r.cost_per_hour * duration_hours for r in resources.values())
        return total_cost
    
    def _estimate_performance(self, resources: Dict[str, Resource], workload: WorkloadRequirement) -> float:
        """Estimate performance for resource allocation."""
        # Simplified performance estimation based on resource adequacy
        performance_score = 1.0
        
        for resource_type, required_amount in workload.resource_requirements.items():
            allocated_amount = sum(
                r.capacity - r.current_usage for r in resources.values() 
                if r.resource_type == resource_type
            )
            
            if allocated_amount > 0:
                adequacy_ratio = min(1.0, allocated_amount / required_amount)
                performance_score *= adequacy_ratio
            else:
                performance_score = 0.0
                break
        
        return performance_score
    
    async def _optimize_greedy(
        self,
        workloads: List[WorkloadRequirement],
        available_resources: Dict[str, Resource],
        carbon_forecasts: Dict[str, CarbonForecast]
    ) -> List[AllocationDecision]:
        """Greedy optimization: minimize carbon intensity first."""
        decisions = []
        
        # Sort workloads by priority
        sorted_workloads = sorted(workloads, key=lambda w: w.priority, reverse=True)
        
        for workload in sorted_workloads:
            # Find lowest carbon time window
            best_allocation = None
            lowest_carbon = float('inf')
            
            candidates = self._generate_candidate_allocations(workload, available_resources, carbon_forecasts)
            
            for candidate in candidates:
                if candidate.expected_carbon_kg < lowest_carbon:
                    lowest_carbon = candidate.expected_carbon_kg
                    best_allocation = candidate
            
            if best_allocation:
                decisions.append(best_allocation)
                # Update resource availability (simplified)
                for resource in best_allocation.allocated_resources.values():
                    resource.current_usage += workload.resource_requirements.get(resource.resource_type, 0)
        
        return decisions
    
    async def _optimize_adaptive_threshold(
        self,
        workloads: List[WorkloadRequirement],
        available_resources: Dict[str, Resource],
        carbon_forecasts: Dict[str, CarbonForecast]
    ) -> List[AllocationDecision]:
        """Adaptive threshold optimization: adjust thresholds based on conditions."""
        decisions = []
        
        # Calculate dynamic carbon threshold
        if carbon_forecasts:
            all_intensities = []
            for forecast in carbon_forecasts.values():
                all_intensities.extend([ci.carbon_intensity for ci in forecast.data_points])
            
            if all_intensities:
                mean_intensity = sum(all_intensities) / len(all_intensities)
                std_intensity = math.sqrt(sum((ci - mean_intensity) ** 2 for ci in all_intensities) / len(all_intensities))
                carbon_threshold = mean_intensity - 0.5 * std_intensity  # Schedule below average - 0.5 std
            else:
                carbon_threshold = 100.0
        else:
            carbon_threshold = 100.0
        
        logger.info(f"Adaptive carbon threshold: {carbon_threshold:.1f} gCO2/kWh")
        
        for workload in workloads:
            candidates = self._generate_candidate_allocations(workload, available_resources, carbon_forecasts)
            
            # Filter candidates by threshold
            valid_candidates = [c for c in candidates if c.expected_carbon_kg <= carbon_threshold * workload.estimated_duration_hours / 100.0]
            
            if valid_candidates:
                # Select best valid candidate (minimize cost among low-carbon options)
                best_candidate = min(valid_candidates, key=lambda c: c.expected_cost_usd)
                decisions.append(best_candidate)
            else:
                # If no candidates meet threshold, select lowest carbon option
                if candidates:
                    best_candidate = min(candidates, key=lambda c: c.expected_carbon_kg)
                    decisions.append(best_candidate)
        
        return decisions
    
    def _convert_rl_result_to_allocations(
        self,
        rl_result: Dict[str, Any],
        workloads: List[WorkloadRequirement],
        available_resources: Dict[str, Resource]
    ) -> List[AllocationDecision]:
        """Convert RL optimization result to allocation decisions."""
        decisions = []
        
        # Extract actions from RL result
        actions = rl_result.get("actions", [])
        
        for i, workload in enumerate(workloads):
            if i < len(actions):
                action = actions[i]["action"]
                
                # Convert action to allocation decision
                if action["action"] == "schedule_now":
                    start_time = datetime.now()
                else:
                    delay_hours = action.get("delay_hours", 1)
                    start_time = datetime.now() + timedelta(hours=delay_hours)
                
                end_time = start_time + timedelta(hours=workload.estimated_duration_hours)
                
                # Simple resource allocation (would be more sophisticated)
                allocated_resources = {}
                for resource_type, required_amount in workload.resource_requirements.items():
                    for resource_id, resource in available_resources.items():
                        if resource.resource_type == resource_type:
                            allocated_resources[resource_id] = resource
                            break
                
                decision = AllocationDecision(
                    workload_id=workload.workload_id,
                    allocated_resources=allocated_resources,
                    start_time=start_time,
                    end_time=end_time,
                    expected_carbon_kg=50.0,  # Would estimate properly
                    expected_cost_usd=25.0,   # Would estimate properly
                    expected_performance=0.9,
                    confidence_score=0.7
                )
                
                decisions.append(decision)
        
        return decisions
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization performance report."""
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "total_optimizations": len(self.optimization_history),
            "strategy_distribution": {},
            "performance_metrics": {},
            "trends": {}
        }
        
        # Strategy distribution
        strategies = [opt["strategy"] for opt in self.optimization_history]
        for strategy in set(strategies):
            report["strategy_distribution"][strategy] = strategies.count(strategy)
        
        # Performance metrics
        optimization_times = [opt["optimization_time_seconds"] for opt in self.optimization_history]
        success_rate = sum(1 for opt in self.optimization_history if opt["success"]) / len(self.optimization_history)
        
        report["performance_metrics"] = {
            "average_optimization_time_seconds": sum(optimization_times) / len(optimization_times),
            "min_optimization_time_seconds": min(optimization_times),
            "max_optimization_time_seconds": max(optimization_times),
            "success_rate": success_rate,
            "total_workloads_optimized": sum(opt["workloads_count"] for opt in self.optimization_history),
            "total_decisions_generated": sum(opt["decisions_count"] for opt in self.optimization_history)
        }
        
        # Recent trends (last 10 optimizations)
        recent_history = self.optimization_history[-10:]
        if len(recent_history) >= 2:
            recent_times = [opt["optimization_time_seconds"] for opt in recent_history]
            recent_success_rate = sum(1 for opt in recent_history if opt["success"]) / len(recent_history)
            
            report["trends"] = {
                "recent_average_time": sum(recent_times) / len(recent_times),
                "recent_success_rate": recent_success_rate,
                "performance_trend": "improving" if recent_times[-1] < recent_times[0] else "stable",
                "optimization_frequency": len(recent_history)
            }
        
        return report