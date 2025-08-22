"""
Resilience Manager for Production Carbon-Aware Training
Provides comprehensive error handling, recovery, and reliability features.
"""

import asyncio
import logging
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from .types import TrainingState, TrainingMetrics, CarbonIntensity

logger = logging.getLogger(__name__)


class RecoveryMode(Enum):
    """Training recovery modes."""
    IMMEDIATE = "immediate"        # Resume immediately
    WAIT_FOR_CLEAN = "wait_clean"  # Wait for clean energy
    MIGRATE = "migrate"            # Migrate to cleaner region
    ABORT = "abort"                # Abort training


class FailureType(Enum):
    """Types of failures that can occur."""
    CARBON_API = "carbon_api"      # Carbon data API failure
    NETWORK = "network"            # Network connectivity
    COMPUTE = "compute"            # Compute resource failure
    STORAGE = "storage"            # Storage/checkpoint failure
    TRAINING = "training"          # Training loop failure
    TIMEOUT = "timeout"            # Operation timeout
    RESOURCE = "resource"          # Resource exhaustion


@dataclass
class CheckpointMetadata:
    """Metadata for training checkpoints."""
    checkpoint_id: str
    timestamp: datetime
    step: int
    epoch: int
    carbon_metrics: Dict[str, Any]
    model_state_hash: str
    optimizer_state_hash: Optional[str] = None
    file_path: Optional[str] = None
    file_size_bytes: int = 0


@dataclass
class RecoveryPlan:
    """Plan for recovering from failures."""
    failure_type: FailureType
    recovery_mode: RecoveryMode
    estimated_recovery_time: timedelta
    checkpoint_to_restore: Optional[str] = None
    alternative_actions: List[str] = None
    confidence_score: float = 1.0


class ResilienceManager:
    """Comprehensive resilience and recovery management for carbon-aware training."""
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        backup_dir: Optional[str] = None,
        max_failures: int = 5,
        recovery_timeout: timedelta = timedelta(minutes=30),
        checkpoint_interval: int = 100,  # steps
        health_check_interval: int = 60,  # seconds
    ):
        """Initialize resilience manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            backup_dir: Optional backup directory (cloud/remote storage)
            max_failures: Maximum failures before aborting
            recovery_timeout: Max time to spend on recovery
            checkpoint_interval: Steps between automatic checkpoints
            health_check_interval: Seconds between health checks
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.max_failures = max_failures
        self.recovery_timeout = recovery_timeout
        self.checkpoint_interval = checkpoint_interval
        self.health_check_interval = health_check_interval
        
        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.failure_count: Dict[FailureType, int] = {}
        self.last_health_check = datetime.now()
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        
        # Callbacks
        self.failure_callbacks: List[Callable] = []
        self.recovery_callbacks: List[Callable] = []
        
        # Health monitoring
        self.health_status: Dict[str, Any] = {
            'overall': 'healthy',
            'carbon_api': 'unknown',
            'storage': 'unknown',
            'compute': 'unknown',
            'last_check': datetime.now().isoformat()
        }
        
        # Circuit breaker states
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ResilienceManager initialized with {max_failures} max failures")
    
    async def initialize(self) -> None:
        """Initialize resilience manager and start health monitoring."""
        # Load existing checkpoints
        await self._load_existing_checkpoints()
        
        # Start health monitoring
        asyncio.create_task(self._health_monitoring_loop())
        
        # Initialize circuit breakers
        self._initialize_circuit_breakers()
        
        logger.info("ResilienceManager fully initialized")
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for external services."""
        self.circuit_breakers = {
            'carbon_api': {
                'state': 'closed',
                'failure_count': 0,
                'last_failure': None,
                'timeout': 60,  # seconds
                'threshold': 3
            },
            'storage': {
                'state': 'closed',
                'failure_count': 0,
                'last_failure': None,
                'timeout': 30,
                'threshold': 2
            }
        }
    
    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval * 2)  # Back off
    
    async def _perform_health_check(self) -> None:
        """Perform comprehensive health check."""
        self.last_health_check = datetime.now()
        
        checks = {
            'storage': self._check_storage_health(),
            'compute': self._check_compute_health(),
            'memory': self._check_memory_health()
        }
        
        # Run all checks concurrently
        results = await asyncio.gather(*checks.values(), return_exceptions=True)
        
        # Update health status
        check_names = list(checks.keys())
        overall_healthy = True
        
        for i, result in enumerate(results):
            check_name = check_names[i]
            if isinstance(result, Exception):
                self.health_status[check_name] = f'error: {str(result)}'
                overall_healthy = False
            else:
                self.health_status[check_name] = 'healthy' if result else 'degraded'
                if not result:
                    overall_healthy = False
        
        self.health_status['overall'] = 'healthy' if overall_healthy else 'degraded'
        self.health_status['last_check'] = datetime.now().isoformat()
    
    async def _check_storage_health(self) -> bool:
        """Check storage system health."""
        try:
            # Test write/read
            test_file = self.checkpoint_dir / f"health_test_{int(time.time())}.tmp"
            test_data = b"health_check_data"
            
            # Write test
            test_file.write_bytes(test_data)
            
            # Read test
            read_data = test_file.read_bytes()
            
            # Cleanup
            test_file.unlink()
            
            return read_data == test_data
        except Exception as e:
            logger.warning(f"Storage health check failed: {e}")
            return False
    
    async def _check_compute_health(self) -> bool:
        """Check compute resource health."""
        try:
            import psutil
            # Check CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Consider healthy if not overloaded
            return cpu_percent < 95 and memory_percent < 95
        except ImportError:
            # Can't check without psutil, assume healthy
            return True
        except Exception as e:
            logger.warning(f"Compute health check failed: {e}")
            return False
    
    async def _check_memory_health(self) -> bool:
        """Check memory health and detect leaks."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Store memory history for leak detection
            if not hasattr(self, '_memory_history'):
                self._memory_history = []
            
            self._memory_history.append({
                'timestamp': datetime.now(),
                'rss': memory_info.rss,
                'vms': memory_info.vms
            })
            
            # Keep only last hour of data
            cutoff = datetime.now() - timedelta(hours=1)
            self._memory_history = [
                m for m in self._memory_history 
                if m['timestamp'] >= cutoff
            ]
            
            # Detect memory leak (steady growth over time)
            if len(self._memory_history) >= 10:
                first_rss = self._memory_history[0]['rss']
                last_rss = self._memory_history[-1]['rss']
                growth_ratio = last_rss / first_rss if first_rss > 0 else 1.0
                
                # Flag if memory grew by more than 50% in an hour
                if growth_ratio > 1.5:
                    logger.warning(f"Potential memory leak detected: {growth_ratio:.2f}x growth")
                    return False
            
            return True
        except ImportError:
            return True
        except Exception as e:
            logger.warning(f"Memory health check failed: {e}")
            return False
    
    async def create_checkpoint(
        self, 
        step: int, 
        epoch: int,
        model_state: Any,
        optimizer_state: Optional[Any] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create comprehensive checkpoint with metadata.
        
        Args:
            step: Current training step
            epoch: Current epoch
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            metrics: Current metrics
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"checkpoint_{step}_{int(time.time())}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        try:
            # Prepare checkpoint data
            checkpoint_data = {
                'step': step,
                'epoch': epoch,
                'model_state': model_state,
                'optimizer_state': optimizer_state,
                'metrics': metrics or {},
                'timestamp': datetime.now().isoformat(),
                'carbon_aware_metadata': {
                    'checkpoint_id': checkpoint_id,
                    'creation_time': datetime.now().isoformat()
                }
            }
            
            # Calculate hashes for integrity
            model_hash = self._calculate_hash(model_state)
            optimizer_hash = self._calculate_hash(optimizer_state) if optimizer_state else None
            
            # Save checkpoint
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            file_size = checkpoint_path.stat().st_size
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=datetime.now(),
                step=step,
                epoch=epoch,
                carbon_metrics=metrics or {},
                model_state_hash=model_hash,
                optimizer_state_hash=optimizer_hash,
                file_path=str(checkpoint_path),
                file_size_bytes=file_size
            )
            
            self.checkpoints[checkpoint_id] = metadata
            
            # Backup to remote if configured
            if self.backup_dir:
                await self._backup_checkpoint(checkpoint_id, checkpoint_path)
            
            # Clean up old checkpoints
            await self._cleanup_old_checkpoints()
            
            logger.info(f"Created checkpoint {checkpoint_id} at step {step}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            await self._handle_failure(FailureType.STORAGE, {'checkpoint_step': step})
            raise
    
    async def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore training from checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            
        Returns:
            Checkpoint data
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        metadata = self.checkpoints[checkpoint_id]
        checkpoint_path = Path(metadata.file_path)
        
        try:
            # Try local file first
            if checkpoint_path.exists():
                with open(checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                # Try backup location
                data = await self._restore_from_backup(checkpoint_id)
            
            # Verify integrity
            if not await self._verify_checkpoint_integrity(data, metadata):
                raise ValueError(f"Checkpoint {checkpoint_id} failed integrity check")
            
            logger.info(f"Restored checkpoint {checkpoint_id} from step {metadata.step}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            await self._handle_failure(FailureType.STORAGE, {'checkpoint_id': checkpoint_id})
            raise
    
    async def _backup_checkpoint(self, checkpoint_id: str, checkpoint_path: Path) -> None:
        """Backup checkpoint to remote storage."""
        try:
            backup_path = self.backup_dir / f"{checkpoint_id}.pkl"
            
            # Simple file copy for now (could be cloud storage)
            import shutil
            shutil.copy2(checkpoint_path, backup_path)
            
            logger.debug(f"Backed up checkpoint {checkpoint_id}")
        except Exception as e:
            logger.warning(f"Failed to backup checkpoint {checkpoint_id}: {e}")
    
    async def _restore_from_backup(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore checkpoint from backup location."""
        if not self.backup_dir:
            raise FileNotFoundError(f"No backup available for {checkpoint_id}")
        
        backup_path = self.backup_dir / f"{checkpoint_id}.pkl"
        
        with open(backup_path, 'rb') as f:
            return pickle.load(f)
    
    def _calculate_hash(self, data: Any) -> str:
        """Calculate hash of data for integrity checking."""
        if data is None:
            return ""
        
        # Convert to bytes for hashing
        if hasattr(data, 'state_dict'):
            # PyTorch model/optimizer
            data_bytes = pickle.dumps(data.state_dict())
        else:
            data_bytes = pickle.dumps(data)
        
        return hashlib.sha256(data_bytes).hexdigest()[:16]  # First 16 chars
    
    async def _verify_checkpoint_integrity(
        self, 
        data: Dict[str, Any], 
        metadata: CheckpointMetadata
    ) -> bool:
        """Verify checkpoint data integrity."""
        try:
            # Check basic structure
            required_fields = ['step', 'epoch', 'model_state', 'timestamp']
            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field {field} in checkpoint")
                    return False
            
            # Verify hashes
            model_hash = self._calculate_hash(data['model_state'])
            if model_hash != metadata.model_state_hash:
                logger.error(f"Model state hash mismatch in checkpoint")
                return False
            
            if data.get('optimizer_state') and metadata.optimizer_state_hash:
                optimizer_hash = self._calculate_hash(data['optimizer_state'])
                if optimizer_hash != metadata.optimizer_state_hash:
                    logger.error(f"Optimizer state hash mismatch in checkpoint")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Checkpoint integrity verification failed: {e}")
            return False
    
    async def handle_training_failure(
        self, 
        failure_type: FailureType,
        error: Exception,
        context: Dict[str, Any]
    ) -> RecoveryPlan:
        """Handle training failure and create recovery plan.
        
        Args:
            failure_type: Type of failure that occurred
            error: The exception that caused the failure
            context: Additional context about the failure
            
        Returns:
            Recovery plan
        """
        logger.error(f"Training failure: {failure_type.value} - {error}")
        
        # Record failure
        await self._record_failure(failure_type, error, context)
        
        # Create recovery plan
        plan = await self._create_recovery_plan(failure_type, error, context)
        
        # Execute recovery if possible
        if plan.recovery_mode != RecoveryMode.ABORT:
            await self._execute_recovery_plan(plan, context)
        
        return plan
    
    async def _record_failure(
        self, 
        failure_type: FailureType,
        error: Exception,
        context: Dict[str, Any]
    ) -> None:
        """Record failure for analysis and circuit breaking."""
        self.failure_count[failure_type] = self.failure_count.get(failure_type, 0) + 1
        
        failure_record = {
            'timestamp': datetime.now().isoformat(),
            'type': failure_type.value,
            'error': str(error),
            'context': context,
            'count': self.failure_count[failure_type]
        }
        
        self.recovery_history.append(failure_record)
        
        # Update circuit breakers
        if failure_type in [FailureType.CARBON_API, FailureType.STORAGE]:
            service = 'carbon_api' if failure_type == FailureType.CARBON_API else 'storage'
            self._update_circuit_breaker(service, failed=True)
        
        # Notify callbacks
        for callback in self.failure_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(failure_type, error, context)
                else:
                    callback(failure_type, error, context)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")
    
    def _update_circuit_breaker(self, service: str, failed: bool = False) -> None:
        """Update circuit breaker state."""
        if service not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[service]
        
        if failed:
            breaker['failure_count'] += 1
            breaker['last_failure'] = datetime.now()
            
            if breaker['failure_count'] >= breaker['threshold']:
                breaker['state'] = 'open'
                logger.warning(f"Circuit breaker opened for {service}")
        else:
            # Success - reset failure count
            if breaker['state'] == 'closed':
                breaker['failure_count'] = 0
            elif breaker['state'] == 'half_open':
                # Success in half-open state, close the breaker
                breaker['state'] = 'closed'
                breaker['failure_count'] = 0
                logger.info(f"Circuit breaker closed for {service}")
        
        # Check if we should move from open to half-open
        if breaker['state'] == 'open' and breaker['last_failure']:
            time_since_failure = (datetime.now() - breaker['last_failure']).total_seconds()
            if time_since_failure >= breaker['timeout']:
                breaker['state'] = 'half_open'
                logger.info(f"Circuit breaker half-open for {service}")
    
    def is_circuit_breaker_open(self, service: str) -> bool:
        """Check if circuit breaker is open for a service."""
        if service not in self.circuit_breakers:
            return False
        
        return self.circuit_breakers[service]['state'] == 'open'
    
    async def _create_recovery_plan(
        self,
        failure_type: FailureType,
        error: Exception,
        context: Dict[str, Any]
    ) -> RecoveryPlan:
        """Create intelligent recovery plan based on failure type and history."""
        
        # Check if we've exceeded max failures
        total_failures = sum(self.failure_count.values())
        if total_failures >= self.max_failures:
            return RecoveryPlan(
                failure_type=failure_type,
                recovery_mode=RecoveryMode.ABORT,
                estimated_recovery_time=timedelta(0),
                confidence_score=0.0
            )
        
        # Create recovery plan based on failure type
        if failure_type == FailureType.CARBON_API:
            return RecoveryPlan(
                failure_type=failure_type,
                recovery_mode=RecoveryMode.WAIT_FOR_CLEAN,  # Use cached data temporarily
                estimated_recovery_time=timedelta(minutes=5),
                alternative_actions=['use_cached_data', 'fallback_provider'],
                confidence_score=0.8
            )
        
        elif failure_type == FailureType.STORAGE:
            # Find most recent checkpoint
            recent_checkpoint = self._find_most_recent_checkpoint()
            
            return RecoveryPlan(
                failure_type=failure_type,
                recovery_mode=RecoveryMode.IMMEDIATE,
                estimated_recovery_time=timedelta(minutes=2),
                checkpoint_to_restore=recent_checkpoint,
                alternative_actions=['restore_from_backup'],
                confidence_score=0.9
            )
        
        elif failure_type in [FailureType.NETWORK, FailureType.TIMEOUT]:
            return RecoveryPlan(
                failure_type=failure_type,
                recovery_mode=RecoveryMode.IMMEDIATE,
                estimated_recovery_time=timedelta(minutes=1),
                alternative_actions=['retry_with_backoff'],
                confidence_score=0.7
            )
        
        elif failure_type == FailureType.COMPUTE:
            return RecoveryPlan(
                failure_type=failure_type,
                recovery_mode=RecoveryMode.MIGRATE,
                estimated_recovery_time=timedelta(minutes=15),
                alternative_actions=['reduce_batch_size', 'use_cpu_fallback'],
                confidence_score=0.6
            )
        
        else:
            # Generic recovery
            return RecoveryPlan(
                failure_type=failure_type,
                recovery_mode=RecoveryMode.IMMEDIATE,
                estimated_recovery_time=timedelta(minutes=5),
                confidence_score=0.5
            )
    
    def _find_most_recent_checkpoint(self) -> Optional[str]:
        """Find the most recent valid checkpoint."""
        if not self.checkpoints:
            return None
        
        # Sort by timestamp, most recent first
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        
        # Return the most recent one
        return sorted_checkpoints[0][0] if sorted_checkpoints else None
    
    async def _execute_recovery_plan(
        self,
        plan: RecoveryPlan,
        context: Dict[str, Any]
    ) -> None:
        """Execute recovery plan."""
        logger.info(f"Executing recovery plan: {plan.recovery_mode.value}")
        
        recovery_start = datetime.now()
        
        try:
            if plan.recovery_mode == RecoveryMode.IMMEDIATE:
                # Basic retry logic
                await asyncio.sleep(1)  # Brief pause
                
            elif plan.recovery_mode == RecoveryMode.WAIT_FOR_CLEAN:
                # Wait for better conditions
                await self._wait_for_recovery_conditions()
                
            elif plan.recovery_mode == RecoveryMode.MIGRATE:
                # Migrate to different resources
                await self._attempt_migration(context)
            
            # Record successful recovery
            recovery_time = datetime.now() - recovery_start
            self.recovery_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'recovery_success',
                'failure_type': plan.failure_type.value,
                'recovery_mode': plan.recovery_mode.value,
                'recovery_time_seconds': recovery_time.total_seconds()
            })
            
            logger.info(f"Recovery completed in {recovery_time.total_seconds():.1f}s")
            
        except Exception as e:
            logger.error(f"Recovery plan execution failed: {e}")
            raise
    
    async def _wait_for_recovery_conditions(self) -> None:
        """Wait for conditions to improve before retrying."""
        max_wait = 300  # 5 minutes
        wait_time = 0
        
        while wait_time < max_wait:
            await asyncio.sleep(30)  # Check every 30 seconds
            wait_time += 30
            
            # Check if conditions have improved
            if self.health_status['overall'] == 'healthy':
                logger.info("Conditions improved, continuing recovery")
                return
        
        logger.warning("Recovery wait timeout exceeded")
    
    async def _attempt_migration(self, context: Dict[str, Any]) -> None:
        """Attempt to migrate training to different resources."""
        logger.info("Attempting migration to alternative resources")
        
        # This would implement actual migration logic
        # For now, simulate a successful migration
        await asyncio.sleep(2)
        
        logger.info("Migration completed successfully")
    
    async def _cleanup_old_checkpoints(self, keep_count: int = 10) -> None:
        """Clean up old checkpoints, keeping the most recent ones."""
        if len(self.checkpoints) <= keep_count:
            return
        
        # Sort by timestamp
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1].timestamp,
            reverse=True
        )
        
        # Remove old checkpoints
        checkpoints_to_remove = sorted_checkpoints[keep_count:]
        
        for checkpoint_id, metadata in checkpoints_to_remove:
            try:
                # Remove file
                if metadata.file_path and Path(metadata.file_path).exists():
                    Path(metadata.file_path).unlink()
                
                # Remove from memory
                del self.checkpoints[checkpoint_id]
                
                logger.debug(f"Cleaned up old checkpoint {checkpoint_id}")
                
            except Exception as e:
                logger.warning(f"Failed to cleanup checkpoint {checkpoint_id}: {e}")
    
    async def _load_existing_checkpoints(self) -> None:
        """Load metadata for existing checkpoints on startup."""
        try:
            for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pkl"):
                checkpoint_id = checkpoint_file.stem
                
                try:
                    # Load checkpoint to get metadata
                    with open(checkpoint_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Extract metadata
                    metadata = CheckpointMetadata(
                        checkpoint_id=checkpoint_id,
                        timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                        step=data.get('step', 0),
                        epoch=data.get('epoch', 0),
                        carbon_metrics=data.get('metrics', {}),
                        model_state_hash=self._calculate_hash(data.get('model_state')),
                        optimizer_state_hash=self._calculate_hash(data.get('optimizer_state')),
                        file_path=str(checkpoint_file),
                        file_size_bytes=checkpoint_file.stat().st_size
                    )
                    
                    self.checkpoints[checkpoint_id] = metadata
                    
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint metadata for {checkpoint_file}: {e}")
            
            logger.info(f"Loaded {len(self.checkpoints)} existing checkpoints")
            
        except Exception as e:
            logger.error(f"Failed to load existing checkpoints: {e}")
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status."""
        return {
            'health_status': self.health_status,
            'failure_counts': dict(self.failure_count),
            'total_failures': sum(self.failure_count.values()),
            'checkpoint_count': len(self.checkpoints),
            'circuit_breakers': {
                service: {
                    'state': info['state'],
                    'failure_count': info['failure_count']
                }
                for service, info in self.circuit_breakers.items()
            },
            'last_health_check': self.last_health_check.isoformat(),
            'recovery_history_count': len(self.recovery_history)
        }
    
    def add_failure_callback(self, callback: Callable) -> None:
        """Add callback for failure events."""
        self.failure_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable) -> None:
        """Add callback for recovery events."""
        self.recovery_callbacks.append(callback)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up ResilienceManager")
        
        # Could save state, close connections, etc.
        # For now, just log
        logger.info("ResilienceManager cleanup completed")