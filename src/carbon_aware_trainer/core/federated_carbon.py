"""
Federated learning with carbon-aware client selection and aggregation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import random

import numpy as np
from pydantic import BaseModel

from .types import CarbonIntensity, CarbonForecast
from .monitor import CarbonMonitor
from .exceptions import CarbonAwareException


logger = logging.getLogger(__name__)


class ClientStatus(Enum):
    """Status of federated learning clients."""
    AVAILABLE = "available"
    TRAINING = "training"
    UPLOADING = "uploading"
    OFFLINE = "offline"
    HIGH_CARBON = "high_carbon"


class AggregationStrategy(Enum):
    """Strategies for carbon-aware aggregation."""
    IMMEDIATE = "immediate"          # Aggregate as soon as updates arrive
    CARBON_OPTIMAL = "carbon_optimal" # Wait for low-carbon periods
    BATCH_OPTIMAL = "batch_optimal"   # Wait for optimal batch of updates
    DEADLINE_AWARE = "deadline_aware" # Balance carbon and deadlines


@dataclass
class FederatedClient:
    """Federated learning client configuration."""
    client_id: str
    region: str
    compute_capacity: float  # Relative capacity (0-1)
    data_samples: int
    carbon_threshold: float = 100.0  # gCO2/kWh
    last_seen: Optional[datetime] = None
    current_carbon: Optional[float] = None
    status: ClientStatus = ClientStatus.AVAILABLE


class ClientSelection(BaseModel):
    """Result of carbon-aware client selection."""
    selected_clients: List[str]
    rejected_clients: Dict[str, str]  # client_id -> reason
    expected_carbon_total: float
    selection_criteria: Dict[str, Any]
    estimated_training_time: float


class CarbonAwareFederated:
    """Carbon-aware federated learning orchestrator."""
    
    def __init__(
        self,
        clients: Dict[str, Dict[str, Any]],
        aggregation_server_region: str = "US-CA",
        monitor: Optional[CarbonMonitor] = None,
        carbon_threshold: float = 100.0,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.CARBON_OPTIMAL
    ):
        self.clients = {
            cid: FederatedClient(**config) if isinstance(config, dict) else config
            for cid, config in clients.items()
        }
        self.aggregation_server_region = aggregation_server_region
        self.monitor = monitor or CarbonMonitor()
        self.carbon_threshold = carbon_threshold
        self.aggregation_strategy = aggregation_strategy
        
        # Federated learning state
        self.current_round = 0
        self.pending_updates = {}  # client_id -> update_data
        self.round_history = []
        self.global_model_version = 0
        
        # Carbon tracking
        self.client_carbon_history = {cid: [] for cid in clients.keys()}
        self.aggregation_carbon_history = []
        
    async def select_clients(
        self,
        round_num: int,
        num_clients: int,
        min_data_samples: int = 100,
        diversity_weight: float = 0.3
    ) -> ClientSelection:
        """Select clients based on carbon intensity and other criteria."""
        logger.info(f"Selecting {num_clients} clients for round {round_num}")
        
        # Get current carbon intensity for all client regions
        await self._update_client_carbon_data()
        
        # Filter available clients
        available_clients = [
            client for client in self.clients.values()
            if (client.status == ClientStatus.AVAILABLE and 
                client.data_samples >= min_data_samples and
                client.current_carbon is not None)
        ]
        
        if len(available_clients) < num_clients:
            logger.warning(f"Only {len(available_clients)} clients available, need {num_clients}")
            
        # Score clients based on multiple criteria
        client_scores = []
        for client in available_clients:
            score = await self._score_client_for_selection(client, diversity_weight)
            client_scores.append((client, score))
            
        # Sort by score (lower is better) and select top clients
        client_scores.sort(key=lambda x: x[1])
        selected_clients = [c[0].client_id for c in client_scores[:num_clients]]
        
        # Calculate rejection reasons
        rejected_clients = {}
        all_client_ids = set(self.clients.keys())
        for cid in all_client_ids - set(selected_clients):
            client = self.clients[cid]
            if client.status != ClientStatus.AVAILABLE:
                rejected_clients[cid] = f"Status: {client.status.value}"
            elif client.data_samples < min_data_samples:
                rejected_clients[cid] = f"Insufficient data: {client.data_samples} < {min_data_samples}"
            elif client.current_carbon and client.current_carbon > client.carbon_threshold:
                rejected_clients[cid] = f"High carbon: {client.current_carbon:.1f} > {client.carbon_threshold}"
            else:
                rejected_clients[cid] = "Not in top selection"
                
        # Calculate expected metrics
        selected_client_objs = [self.clients[cid] for cid in selected_clients]
        expected_carbon = sum(c.current_carbon or 0 for c in selected_client_objs)
        avg_capacity = np.mean([c.compute_capacity for c in selected_client_objs])
        estimated_time = 60 / avg_capacity if avg_capacity > 0 else 120  # minutes
        
        selection = ClientSelection(
            selected_clients=selected_clients,
            rejected_clients=rejected_clients,
            expected_carbon_total=expected_carbon,
            selection_criteria={
                "carbon_threshold": self.carbon_threshold,
                "min_data_samples": min_data_samples,
                "diversity_weight": diversity_weight,
                "available_clients": len(available_clients)
            },
            estimated_training_time=estimated_time
        )
        
        return selection
        
    async def _update_client_carbon_data(self) -> None:
        """Update carbon intensity data for all clients."""
        tasks = []
        for client in self.clients.values():
            task = self._get_client_carbon(client)
            tasks.append(task)
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _get_client_carbon(self, client: FederatedClient) -> None:
        """Get current carbon intensity for a client."""
        try:
            carbon_data = await self.monitor.get_current_intensity(client.region)
            if carbon_data:
                client.current_carbon = carbon_data.carbon_intensity
                client.last_seen = datetime.now()
                
                # Update client status based on carbon
                if client.current_carbon > client.carbon_threshold:
                    client.status = ClientStatus.HIGH_CARBON
                elif client.status == ClientStatus.HIGH_CARBON:
                    client.status = ClientStatus.AVAILABLE
                    
                # Track carbon history
                self.client_carbon_history[client.client_id].append({
                    'timestamp': datetime.now(),
                    'carbon_intensity': client.current_carbon
                })
                
        except Exception as e:
            logger.warning(f"Failed to get carbon data for client {client.client_id}: {e}")
            client.status = ClientStatus.OFFLINE
            
    async def _score_client_for_selection(
        self,
        client: FederatedClient,
        diversity_weight: float
    ) -> float:
        """Score a client for selection (lower is better)."""
        score = 0.0
        
        # Carbon score (primary factor)
        if client.current_carbon is not None:
            carbon_score = client.current_carbon / 100.0  # Normalize
            score += carbon_score * 1.0  # Weight of 1.0
            
        # Compute capacity (higher capacity is better, so subtract)
        capacity_score = (1.0 - client.compute_capacity) * 0.5
        score += capacity_score
        
        # Data quantity (more data is better)
        max_samples = max(c.data_samples for c in self.clients.values())
        data_score = (max_samples - client.data_samples) / max_samples * 0.3
        score += data_score
        
        # Diversity bonus (if client region is underrepresented)
        if diversity_weight > 0:
            region_count = sum(1 for c in self.clients.values() if c.region == client.region)
            diversity_penalty = region_count / len(self.clients) * diversity_weight
            score += diversity_penalty
            
        return score
        
    async def train_client(
        self,
        client_id: str,
        model_parameters: Dict[str, Any],
        local_epochs: int = 5,
        batch_size: int = 32
    ) -> Optional[Dict[str, Any]]:
        """Train model on selected client with carbon awareness."""
        client = self.clients.get(client_id)
        if not client:
            raise CarbonAwareException(f"Client {client_id} not found")
            
        if client.status != ClientStatus.AVAILABLE:
            logger.warning(f"Client {client_id} not available: {client.status}")
            return None
            
        logger.info(f"Starting training on client {client_id}")
        client.status = ClientStatus.TRAINING
        
        try:
            # Monitor carbon during training
            training_start = datetime.now()
            carbon_samples = []
            
            # Simulate training with periodic carbon checks
            for epoch in range(local_epochs):
                # Check carbon intensity
                await self._get_client_carbon(client)
                if client.current_carbon:
                    carbon_samples.append(client.current_carbon)
                    
                    # Pause if carbon gets too high
                    if client.current_carbon > client.carbon_threshold * 1.5:
                        logger.info(f"Client {client_id}: High carbon ({client.current_carbon:.1f}), pausing training")
                        await asyncio.sleep(random.uniform(30, 60))  # Pause 30-60 seconds
                        
                # Simulate training time
                training_time = random.uniform(5, 15) / client.compute_capacity
                await asyncio.sleep(min(training_time, 10))  # Cap at 10 seconds for demo
                
            # Simulate model update generation
            client.status = ClientStatus.UPLOADING
            await asyncio.sleep(random.uniform(1, 3))
            
            training_duration = (datetime.now() - training_start).total_seconds() / 60
            avg_carbon = np.mean(carbon_samples) if carbon_samples else 0
            
            # Generate synthetic update
            update = {
                'client_id': client_id,
                'model_update': f"synthetic_update_{client_id}_{self.current_round}",
                'data_samples': client.data_samples,
                'training_duration_minutes': training_duration,
                'average_carbon_intensity': avg_carbon,
                'carbon_samples': carbon_samples,
                'timestamp': datetime.now().isoformat()
            }
            
            client.status = ClientStatus.AVAILABLE
            logger.info(f"Training completed on client {client_id}, avg carbon: {avg_carbon:.1f}")
            
            return update
            
        except Exception as e:
            logger.error(f"Training failed on client {client_id}: {e}")
            client.status = ClientStatus.AVAILABLE
            return None
            
    async def carbon_aware_aggregate(
        self,
        client_updates: List[Dict[str, Any]],
        max_wait_hours: int = 6,
        min_updates: int = 3
    ) -> Dict[str, Any]:
        """Perform carbon-aware aggregation of client updates."""
        logger.info(f"Starting carbon-aware aggregation with {len(client_updates)} updates")
        
        if len(client_updates) < min_updates:
            logger.warning(f"Only {len(client_updates)} updates, minimum is {min_updates}")
            
        # Get aggregation server carbon intensity
        server_carbon = await self.monitor.get_current_intensity(self.aggregation_server_region)
        
        if self.aggregation_strategy == AggregationStrategy.CARBON_OPTIMAL:
            # Wait for low carbon period if current carbon is high
            if server_carbon and server_carbon.carbon_intensity > self.carbon_threshold:
                optimal_window = await self._find_optimal_aggregation_window(max_wait_hours)
                if optimal_window:
                    wait_minutes = optimal_window['wait_minutes']
                    logger.info(f"Waiting {wait_minutes} minutes for better carbon conditions")
                    await asyncio.sleep(min(wait_minutes * 60, 300))  # Cap at 5 minutes for demo
                    
        # Perform aggregation
        aggregation_start = datetime.now()
        
        # Get current server carbon for aggregation
        final_server_carbon = await self.monitor.get_current_intensity(self.aggregation_server_region)
        server_intensity = final_server_carbon.carbon_intensity if final_server_carbon else 0
        
        # Simple federated averaging simulation
        total_samples = sum(update['data_samples'] for update in client_updates)
        weighted_updates = []
        
        for update in client_updates:
            weight = update['data_samples'] / total_samples
            weighted_updates.append({
                'client_id': update['client_id'],
                'weight': weight,
                'carbon_intensity': update.get('average_carbon_intensity', 0)
            })
            
        # Simulate aggregation computation time
        aggregation_time = len(client_updates) * 2  # 2 seconds per update
        await asyncio.sleep(min(aggregation_time, 10))
        
        aggregation_duration = (datetime.now() - aggregation_start).total_seconds() / 60
        
        # Calculate aggregation metrics
        client_carbon_sum = sum(u.get('average_carbon_intensity', 0) * u['data_samples'] for u in client_updates)
        avg_client_carbon = client_carbon_sum / total_samples if total_samples > 0 else 0
        
        self.global_model_version += 1
        
        aggregation_result = {
            'round': self.current_round,
            'global_model_version': self.global_model_version,
            'participating_clients': [u['client_id'] for u in client_updates],
            'client_weights': {u['client_id']: u['weight'] for u in weighted_updates},
            'total_data_samples': total_samples,
            'server_carbon_intensity': server_intensity,
            'average_client_carbon': avg_client_carbon,
            'aggregation_duration_minutes': aggregation_duration,
            'aggregation_timestamp': datetime.now().isoformat(),
            'carbon_savings_estimate': self._estimate_carbon_savings(client_updates, server_intensity)
        }
        
        # Track aggregation carbon
        self.aggregation_carbon_history.append({
            'round': self.current_round,
            'timestamp': datetime.now(),
            'server_carbon': server_intensity,
            'client_carbons': [u.get('average_carbon_intensity', 0) for u in client_updates]
        })
        
        self.round_history.append(aggregation_result)
        self.current_round += 1
        
        logger.info(f"Aggregation completed for round {aggregation_result['round']}")
        return aggregation_result
        
    def _estimate_carbon_savings(
        self,
        client_updates: List[Dict[str, Any]],
        server_carbon: float
    ) -> Dict[str, float]:
        """Estimate carbon savings from carbon-aware selection."""
        # Baseline: if we had used all clients regardless of carbon
        all_carbons = [c.current_carbon or 200 for c in self.clients.values() if c.current_carbon]
        baseline_avg = np.mean(all_carbons) if all_carbons else 200
        
        # Actual: carbon intensity of selected clients
        actual_carbons = [u.get('average_carbon_intensity', 0) for u in client_updates]
        actual_avg = np.mean(actual_carbons) if actual_carbons else 0
        
        client_savings_pct = max(0, (baseline_avg - actual_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
        
        # Server aggregation: vs high-carbon baseline
        server_baseline = 200  # High carbon baseline
        server_savings_pct = max(0, (server_baseline - server_carbon) / server_baseline * 100)
        
        return {
            'client_selection_savings_pct': client_savings_pct,
            'server_aggregation_savings_pct': server_savings_pct,
            'total_estimated_savings_pct': (client_savings_pct + server_savings_pct) / 2
        }
        
    async def _find_optimal_aggregation_window(self, max_wait_hours: int) -> Optional[Dict[str, Any]]:
        """Find optimal window for low-carbon aggregation."""
        try:
            forecast = await self.monitor.get_forecast(self.aggregation_server_region, hours=max_wait_hours)
            
            if not forecast or not forecast.data_points:
                return None
                
            # Find the point with lowest carbon in forecast
            min_carbon_point = min(forecast.data_points, key=lambda dp: dp.carbon_intensity)
            
            if min_carbon_point.carbon_intensity < self.carbon_threshold:
                wait_minutes = (min_carbon_point.timestamp - datetime.now()).total_seconds() / 60
                if 0 < wait_minutes <= max_wait_hours * 60:
                    return {
                        'wait_minutes': wait_minutes,
                        'expected_carbon': min_carbon_point.carbon_intensity,
                        'timestamp': min_carbon_point.timestamp
                    }
                    
        except Exception as e:
            logger.warning(f"Failed to find optimal aggregation window: {e}")
            
        return None
        
    def get_federation_summary(self) -> Dict[str, Any]:
        """Get summary of federated learning progress."""
        active_clients = sum(1 for c in self.clients.values() if c.status != ClientStatus.OFFLINE)
        avg_client_carbon = np.mean([
            c.current_carbon for c in self.clients.values() 
            if c.current_carbon is not None
        ])
        
        return {
            'current_round': self.current_round,
            'global_model_version': self.global_model_version,
            'total_clients': len(self.clients),
            'active_clients': active_clients,
            'average_client_carbon': avg_client_carbon,
            'aggregation_strategy': self.aggregation_strategy.value,
            'carbon_threshold': self.carbon_threshold,
            'total_rounds_completed': len(self.round_history)
        }
        
    def get_client_carbon_history(self, client_id: str) -> List[Dict[str, Any]]:
        """Get carbon history for specific client."""
        return self.client_carbon_history.get(client_id, [])
        
    def get_round_details(self, round_num: int) -> Optional[Dict[str, Any]]:
        """Get details for specific round."""
        for round_data in self.round_history:
            if round_data['round'] == round_num:
                return round_data
        return None