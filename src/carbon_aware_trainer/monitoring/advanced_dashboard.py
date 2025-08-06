"""Advanced real-time dashboard for carbon-aware training."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ..core.types import CarbonIntensity, TrainingMetrics
from ..core.monitor import CarbonMonitor

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics."""
    
    timestamp: datetime
    current_carbon_intensity: float
    avg_carbon_intensity: float
    peak_carbon_intensity: float
    total_emissions_kg: float
    energy_consumption_kwh: float
    training_efficiency: float
    cost_estimate_usd: float
    carbon_savings_kg: float
    renewable_percentage: float
    
    # Training state
    training_state: str
    current_step: int
    current_epoch: int
    pause_duration_minutes: float
    
    # Predictions
    estimated_completion_time: Optional[datetime] = None
    predicted_total_emissions_kg: Optional[float] = None
    confidence_score: float = 0.8


@dataclass
class AlertConfig:
    """Configuration for carbon alerts."""
    
    carbon_threshold_gco2: float = 200
    cost_threshold_usd: float = 100
    pause_duration_threshold_minutes: float = 60
    email_notifications: bool = False
    slack_webhook: Optional[str] = None
    alert_cooldown_minutes: int = 15


class AdvancedCarbonDashboard:
    """Advanced real-time dashboard with predictive analytics."""
    
    def __init__(
        self,
        monitor: CarbonMonitor,
        alert_config: Optional[AlertConfig] = None,
        update_interval_seconds: int = 30,
        history_retention_hours: int = 168  # 1 week
    ):
        """Initialize advanced dashboard.
        
        Args:
            monitor: Carbon monitor instance
            alert_config: Alert configuration
            update_interval_seconds: Dashboard update frequency
            history_retention_hours: How long to keep historical data
        """
        self.monitor = monitor
        self.alert_config = alert_config or AlertConfig()
        self.update_interval = update_interval_seconds
        self.retention_hours = history_retention_hours
        
        # Data storage
        self.metrics_history: List[DashboardMetrics] = []
        self.carbon_forecast: List[CarbonIntensity] = []
        self.training_events: List[Dict[str, Any]] = []
        
        # Dashboard state
        self._dashboard_task: Optional[asyncio.Task] = None
        self._last_alert_time: Dict[str, datetime] = {}
        self._is_running = False
        
        # Predictive models (simplified placeholders)
        self._carbon_predictor = SimpleCarbonPredictor()
        self._cost_estimator = TrainingCostEstimator()
    
    async def start_dashboard(self) -> None:
        """Start the real-time dashboard."""
        if self._is_running:
            logger.warning("Dashboard already running")
            return
        
        self._is_running = True
        self._dashboard_task = asyncio.create_task(self._dashboard_loop())
        logger.info("Advanced carbon dashboard started")
    
    async def stop_dashboard(self) -> None:
        """Stop the dashboard."""
        self._is_running = False
        
        if self._dashboard_task and not self._dashboard_task.done():
            self._dashboard_task.cancel()
            try:
                await self._dashboard_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Advanced carbon dashboard stopped")
    
    async def _dashboard_loop(self) -> None:
        """Main dashboard update loop."""
        while self._is_running:
            try:
                # Collect current metrics
                current_metrics = await self._collect_current_metrics()
                
                if current_metrics:
                    # Add to history
                    self.metrics_history.append(current_metrics)
                    
                    # Cleanup old data
                    self._cleanup_old_data()
                    
                    # Check for alerts
                    await self._check_alerts(current_metrics)
                    
                    # Update forecasts
                    await self._update_forecasts()
                    
                    # Log key metrics
                    self._log_dashboard_summary(current_metrics)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in dashboard loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _collect_current_metrics(self) -> Optional[DashboardMetrics]:
        """Collect current training and carbon metrics."""
        try:
            # Get carbon data from monitor
            regions = self.monitor.regions
            if not regions:
                return None
            
            primary_region = regions[0]
            current_intensity = await self.monitor.get_current_intensity(primary_region)
            
            if not current_intensity:
                return None
            
            # Calculate aggregated metrics from history
            recent_metrics = self._get_recent_metrics(hours=1)
            
            if recent_metrics:
                avg_carbon = sum(m.current_carbon_intensity for m in recent_metrics) / len(recent_metrics)
                peak_carbon = max(m.current_carbon_intensity for m in recent_metrics)
                total_emissions = recent_metrics[-1].total_emissions_kg if recent_metrics else 0
                energy_consumption = recent_metrics[-1].energy_consumption_kwh if recent_metrics else 0
            else:
                avg_carbon = current_intensity.carbon_intensity
                peak_carbon = current_intensity.carbon_intensity
                total_emissions = 0
                energy_consumption = 0
            
            # Estimate costs
            cost_estimate = self._cost_estimator.estimate_current_cost(
                energy_consumption, primary_region
            )
            
            # Calculate carbon savings (vs baseline)
            carbon_savings = self._calculate_carbon_savings(total_emissions)
            
            # Predict completion time
            completion_prediction = self._predict_completion_time()
            
            metrics = DashboardMetrics(
                timestamp=datetime.now(),
                current_carbon_intensity=current_intensity.carbon_intensity,
                avg_carbon_intensity=avg_carbon,
                peak_carbon_intensity=peak_carbon,
                total_emissions_kg=total_emissions,
                energy_consumption_kwh=energy_consumption,
                training_efficiency=self._calculate_training_efficiency(),
                cost_estimate_usd=cost_estimate,
                carbon_savings_kg=carbon_savings,
                renewable_percentage=current_intensity.renewable_percentage or 0,
                training_state="running",  # Would get from actual trainer
                current_step=len(self.metrics_history),  # Simplified
                current_epoch=len(self.metrics_history) // 100,  # Simplified
                pause_duration_minutes=0,  # Would calculate actual pause time
                estimated_completion_time=completion_prediction,
                predicted_total_emissions_kg=self._predict_total_emissions()
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting dashboard metrics: {e}")
            return None
    
    def _get_recent_metrics(self, hours: int = 1) -> List[DashboardMetrics]:
        """Get metrics from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
    
    def _cleanup_old_data(self) -> None:
        """Remove old data beyond retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Clean metrics history
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        # Clean training events
        self.training_events = [
            e for e in self.training_events
            if datetime.fromisoformat(e['timestamp']) >= cutoff_time
        ]
    
    async def _check_alerts(self, metrics: DashboardMetrics) -> None:
        """Check for alert conditions and send notifications."""
        alerts_to_send = []
        
        # High carbon intensity alert
        if metrics.current_carbon_intensity > self.alert_config.carbon_threshold_gco2:
            if self._should_send_alert('high_carbon'):
                alerts_to_send.append({
                    'type': 'high_carbon',
                    'severity': 'warning',
                    'message': f"High carbon intensity: {metrics.current_carbon_intensity:.1f} gCO2/kWh",
                    'data': {'carbon_intensity': metrics.current_carbon_intensity}
                })
        
        # High cost alert
        if metrics.cost_estimate_usd > self.alert_config.cost_threshold_usd:
            if self._should_send_alert('high_cost'):
                alerts_to_send.append({
                    'type': 'high_cost',
                    'severity': 'warning',
                    'message': f"Training cost estimate: ${metrics.cost_estimate_usd:.2f}",
                    'data': {'cost_estimate': metrics.cost_estimate_usd}
                })
        
        # Long pause duration alert
        if metrics.pause_duration_minutes > self.alert_config.pause_duration_threshold_minutes:
            if self._should_send_alert('long_pause'):
                alerts_to_send.append({
                    'type': 'long_pause', 
                    'severity': 'info',
                    'message': f"Training paused for {metrics.pause_duration_minutes:.1f} minutes",
                    'data': {'pause_duration_minutes': metrics.pause_duration_minutes}
                })
        
        # Send alerts
        for alert in alerts_to_send:
            await self._send_alert(alert)
    
    def _should_send_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed since last alert of this type."""
        last_alert = self._last_alert_time.get(alert_type)
        if not last_alert:
            return True
        
        cooldown = timedelta(minutes=self.alert_config.alert_cooldown_minutes)
        return datetime.now() - last_alert > cooldown
    
    async def _send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert notification."""
        # Log alert
        logger.warning(f"ALERT [{alert['type']}]: {alert['message']}")
        
        # Record alert time
        self._last_alert_time[alert['type']] = datetime.now()
        
        # Add to training events
        self.training_events.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'alert',
            'subtype': alert['type'],
            'severity': alert['severity'],
            'message': alert['message'],
            'data': alert['data']
        })
        
        # TODO: Implement actual notification sending
        # - Email notifications
        # - Slack webhooks
        # - SMS alerts
        # - Integration with monitoring systems
    
    async def _update_forecasts(self) -> None:
        """Update carbon intensity forecasts."""
        try:
            # Get forecast for next 48 hours
            primary_region = self.monitor.regions[0] if self.monitor.regions else "US-CA"
            forecast = await self.monitor.get_forecast(primary_region, hours=48)
            
            if forecast and forecast.data_points:
                self.carbon_forecast = forecast.data_points
        
        except Exception as e:
            logger.error(f"Error updating forecasts: {e}")
    
    def _calculate_carbon_savings(self, current_emissions: float) -> float:
        """Calculate carbon savings vs baseline training."""
        # Simplified calculation - would compare against actual baseline
        baseline_intensity = 350  # Assume baseline regional intensity
        current_avg = (
            sum(m.avg_carbon_intensity for m in self.metrics_history[-10:]) / 
            min(10, len(self.metrics_history))
        ) if self.metrics_history else 0
        
        if current_avg > 0:
            savings_rate = (baseline_intensity - current_avg) / baseline_intensity
            return current_emissions * savings_rate
        
        return 0
    
    def _calculate_training_efficiency(self) -> float:
        """Calculate training efficiency score."""
        # Simplified efficiency metric
        if len(self.metrics_history) < 2:
            return 0.8
        
        # Consider pause time impact
        recent_metrics = self._get_recent_metrics(hours=1)
        total_pause_time = sum(m.pause_duration_minutes for m in recent_metrics)
        total_time_minutes = len(recent_metrics) * (self.update_interval / 60)
        
        if total_time_minutes > 0:
            efficiency = 1.0 - (total_pause_time / total_time_minutes)
            return max(0, min(1, efficiency))
        
        return 0.8
    
    def _predict_completion_time(self) -> Optional[datetime]:
        """Predict training completion time."""
        # Simplified prediction based on current progress
        if len(self.metrics_history) < 10:
            return None
        
        # Estimate remaining time based on progress rate
        recent_progress = self.metrics_history[-1].current_step - self.metrics_history[-10].current_step
        if recent_progress <= 0:
            return None
        
        progress_rate = recent_progress / 10  # steps per update
        remaining_steps = 1000 - self.metrics_history[-1].current_step  # Assume 1000 total steps
        
        if progress_rate > 0:
            remaining_updates = remaining_steps / progress_rate
            remaining_time = timedelta(seconds=remaining_updates * self.update_interval)
            return datetime.now() + remaining_time
        
        return None
    
    def _predict_total_emissions(self) -> Optional[float]:
        """Predict total training emissions."""
        return self._carbon_predictor.predict_total_emissions(self.metrics_history)
    
    def _log_dashboard_summary(self, metrics: DashboardMetrics) -> None:
        """Log key dashboard metrics."""
        logger.info(
            f"Carbon Dashboard | "
            f"Intensity: {metrics.current_carbon_intensity:.1f} gCO2/kWh | "
            f"Emissions: {metrics.total_emissions_kg:.2f} kg CO2 | "
            f"Savings: {metrics.carbon_savings_kg:.2f} kg CO2 | "
            f"Cost: ${metrics.cost_estimate_usd:.2f} | "
            f"Efficiency: {metrics.training_efficiency:.1%}"
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data for web interface.
        
        Returns:
            Complete dashboard data structure
        """
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'metrics_history': [asdict(m) for m in self.metrics_history[-100:]],  # Last 100 points
            'carbon_forecast': [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'carbon_intensity': p.carbon_intensity,
                    'renewable_percentage': p.renewable_percentage
                }
                for p in self.carbon_forecast
            ],
            'training_events': self.training_events[-50:],  # Last 50 events
            'summary_stats': self._generate_summary_stats()
        }
    
    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.metrics_history:
            return {}
        
        recent_24h = self._get_recent_metrics(hours=24)
        
        return {
            'total_runtime_hours': len(self.metrics_history) * (self.update_interval / 3600),
            'avg_carbon_intensity_24h': (
                sum(m.current_carbon_intensity for m in recent_24h) / len(recent_24h)
                if recent_24h else 0
            ),
            'peak_carbon_intensity_24h': (
                max(m.current_carbon_intensity for m in recent_24h)
                if recent_24h else 0
            ),
            'total_emissions_kg': recent_24h[-1].total_emissions_kg if recent_24h else 0,
            'total_carbon_savings_kg': recent_24h[-1].carbon_savings_kg if recent_24h else 0,
            'avg_renewable_percentage': (
                sum(m.renewable_percentage for m in recent_24h) / len(recent_24h)
                if recent_24h else 0
            )
        }


class SimpleCarbonPredictor:
    """Simplified carbon intensity predictor."""
    
    def predict_total_emissions(self, metrics_history: List[DashboardMetrics]) -> Optional[float]:
        """Predict total training emissions."""
        if len(metrics_history) < 10:
            return None
        
        # Simple linear extrapolation
        recent_emissions = [m.total_emissions_kg for m in metrics_history[-10:]]
        emission_rate = (recent_emissions[-1] - recent_emissions[0]) / 10
        
        # Assume 1000 total steps, current step from last metric
        current_step = metrics_history[-1].current_step
        remaining_steps = 1000 - current_step
        
        if remaining_steps > 0:
            predicted_additional_emissions = emission_rate * remaining_steps
            return recent_emissions[-1] + predicted_additional_emissions
        
        return recent_emissions[-1]


class TrainingCostEstimator:
    """Training cost estimation."""
    
    def __init__(self):
        # Regional cost per kWh (USD)
        self.regional_electricity_costs = {
            'US-CA': 0.20,
            'US-TX': 0.12,
            'EU-FR': 0.18,
            'EU-DE': 0.30,
            'CN-GD': 0.08
        }
        
        # GPU instance costs per hour (simplified)
        self.gpu_costs_per_hour = {
            'US-CA': 2.50,
            'US-TX': 2.20,
            'EU-FR': 2.80,
            'EU-DE': 3.20,
            'CN-GD': 1.80
        }
    
    def estimate_current_cost(self, energy_kwh: float, region: str) -> float:
        """Estimate current training cost."""
        electricity_rate = self.regional_electricity_costs.get(region, 0.15)
        gpu_rate = self.gpu_costs_per_hour.get(region, 2.50)
        
        # Simplified cost calculation
        electricity_cost = energy_kwh * electricity_rate
        gpu_cost = 1 * gpu_rate  # Assume 1 hour of GPU time for simplicity
        
        return electricity_cost + gpu_cost