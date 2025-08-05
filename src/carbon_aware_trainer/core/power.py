"""Power consumption monitoring and estimation utilities."""

import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import subprocess
import json

from .exceptions import PowerMeteringError, MetricsError


logger = logging.getLogger(__name__)


@dataclass
class PowerReading:
    """Power consumption reading."""
    timestamp: datetime
    device: str
    power_watts: float
    temperature_c: Optional[float] = None
    utilization_percent: Optional[float] = None
    memory_usage_mb: Optional[int] = None


@dataclass
class SystemPowerProfile:
    """System power consumption profile."""
    idle_power_watts: float
    cpu_max_power_watts: float
    gpu_power_watts_per_device: float
    memory_power_watts_per_gb: float
    network_power_watts: float
    storage_power_watts: float


class PowerMonitor:
    """Real-time power consumption monitoring."""
    
    def __init__(self, monitoring_interval: int = 10):
        """Initialize power monitor.
        
        Args:
            monitoring_interval: Monitoring interval in seconds
        """
        self.monitoring_interval = monitoring_interval
        self._readings: List[PowerReading] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = False
        
        # Detect available monitoring tools
        self._nvidia_smi_available = self._check_nvidia_smi()
        self._intel_gpu_top_available = self._check_intel_gpu_top()
        self._cpu_power_available = self._check_cpu_power_monitoring()
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _check_intel_gpu_top(self) -> bool:
        """Check if intel_gpu_top is available."""
        try:
            result = subprocess.run(
                ['intel_gpu_top', '--help'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _check_cpu_power_monitoring(self) -> bool:
        """Check if CPU power monitoring is available."""
        # Check for Intel RAPL (Running Average Power Limit)
        rapl_path = '/sys/class/powercap/intel-rapl'
        if os.path.exists(rapl_path):
            return True
        
        # Check for AMD power monitoring
        amd_energy_path = '/sys/kernel/debug/amd_energy'
        if os.path.exists(amd_energy_path):
            return True
        
        return False
    
    async def start_monitoring(self) -> None:
        """Start continuous power monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Power monitoring already started")
            return
        
        self._stop_monitoring = False
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started power monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop power monitoring."""
        self._stop_monitoring = True
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped power monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_monitoring:
            try:
                readings = await self._collect_power_readings()
                self._readings.extend(readings)
                
                # Keep only recent readings (last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self._readings = [
                    r for r in self._readings 
                    if r.timestamp > cutoff_time
                ]
                
                logger.debug(f"Collected {len(readings)} power readings")
                
            except Exception as e:
                logger.error(f"Error collecting power readings: {e}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_power_readings(self) -> List[PowerReading]:
        """Collect current power readings from available sources."""
        readings = []
        timestamp = datetime.now()
        
        try:
            # GPU power monitoring
            if self._nvidia_smi_available:
                gpu_readings = await self._get_nvidia_gpu_power()
                readings.extend(gpu_readings)
            
            # CPU power monitoring
            if self._cpu_power_available:
                cpu_reading = await self._get_cpu_power()
                if cpu_reading:
                    readings.append(cpu_reading)
            
            # System-level power estimation if direct monitoring not available
            if not readings:
                estimated_reading = await self._estimate_system_power()
                if estimated_reading:
                    readings.append(estimated_reading)
                    
        except Exception as e:
            raise PowerMeteringError("system", e)
        
        return readings
    
    async def _get_nvidia_gpu_power(self) -> List[PowerReading]:
        """Get NVIDIA GPU power consumption."""
        try:
            # Query GPU power using nvidia-smi
            cmd = [
                'nvidia-smi',
                '--query-gpu=timestamp,index,power.draw,temperature.gpu,utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'
            ]
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.warning(f"nvidia-smi error: {stderr.decode()}")
                return []
            
            readings = []
            for line in stdout.decode().strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 6:
                        try:
                            timestamp_str, gpu_id, power_str, temp_str, util_str, mem_str = parts[:6]
                            
                            # Parse values
                            power = float(power_str) if power_str != '[Not Supported]' else None
                            temp = float(temp_str) if temp_str != '[Not Supported]' else None
                            util = float(util_str) if util_str != '[Not Supported]' else None
                            memory = int(float(mem_str)) if mem_str != '[Not Supported]' else None
                            
                            if power is not None:
                                readings.append(PowerReading(
                                    timestamp=datetime.now(),
                                    device=f"nvidia_gpu_{gpu_id}",
                                    power_watts=power,
                                    temperature_c=temp,
                                    utilization_percent=util,
                                    memory_usage_mb=memory
                                ))
                        except ValueError as e:
                            logger.debug(f"Failed to parse nvidia-smi output: {e}")
            
            return readings
            
        except Exception as e:
            logger.warning(f"Failed to get NVIDIA GPU power: {e}")
            return []
    
    async def _get_cpu_power(self) -> Optional[PowerReading]:
        """Get CPU power consumption using RAPL."""
        try:
            # Try Intel RAPL first
            rapl_base = '/sys/class/powercap/intel-rapl'
            
            if os.path.exists(rapl_base):
                total_power = 0.0
                package_count = 0
                
                for package_dir in os.listdir(rapl_base):
                    if package_dir.startswith('intel-rapl:'):
                        energy_file = os.path.join(rapl_base, package_dir, 'energy_uj')
                        if os.path.exists(energy_file):
                            try:
                                with open(energy_file, 'r') as f:
                                    energy_uj = int(f.read().strip())
                                
                                # Convert microjoules to watts (approximate)
                                # This is simplified - real implementation would track energy over time
                                power_estimate = energy_uj / 1_000_000 / self.monitoring_interval
                                total_power += power_estimate
                                package_count += 1
                                
                            except (IOError, ValueError) as e:
                                logger.debug(f"Failed to read RAPL energy: {e}")
                
                if package_count > 0:
                    return PowerReading(
                        timestamp=datetime.now(),
                        device="cpu_rapl",
                        power_watts=total_power
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get CPU power: {e}")
            return None
    
    async def _estimate_system_power(self) -> Optional[PowerReading]:
        """Estimate system power consumption based on utilization."""
        try:
            import psutil
            
            # Get system utilization
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Simple power estimation model
            # These are very rough estimates - real values vary significantly
            base_power = 50  # Idle system power (watts)
            cpu_power = (cpu_percent / 100) * 65  # Max CPU power
            memory_power = (memory.percent / 100) * 10  # Memory power
            
            total_power = base_power + cpu_power + memory_power
            
            return PowerReading(
                timestamp=datetime.now(),
                device="system_estimated",
                power_watts=total_power,
                utilization_percent=cpu_percent
            )
            
        except Exception as e:
            logger.warning(f"Failed to estimate system power: {e}")
            return None
    
    def get_current_power(self) -> Optional[float]:
        """Get current total power consumption.
        
        Returns:
            Current power consumption in watts or None if unavailable
        """
        if not self._readings:
            return None
        
        # Get most recent readings (within last 30 seconds)
        recent_cutoff = datetime.now() - timedelta(seconds=30)
        recent_readings = [
            r for r in self._readings 
            if r.timestamp > recent_cutoff
        ]
        
        if not recent_readings:
            return None
        
        # Sum power from all devices
        total_power = sum(r.power_watts for r in recent_readings)
        return total_power
    
    def get_average_power(self, duration: timedelta = timedelta(minutes=5)) -> Optional[float]:
        """Get average power consumption over a time period.
        
        Args:
            duration: Time period for averaging
            
        Returns:
            Average power consumption in watts
        """
        cutoff_time = datetime.now() - duration
        relevant_readings = [
            r for r in self._readings 
            if r.timestamp > cutoff_time
        ]
        
        if not relevant_readings:
            return None
        
        # Group readings by device and timestamp, then average
        device_powers = {}
        for reading in relevant_readings:
            if reading.device not in device_powers:
                device_powers[reading.device] = []
            device_powers[reading.device].append(reading.power_watts)
        
        # Average each device, then sum
        total_avg_power = 0.0
        for device, powers in device_powers.items():
            avg_device_power = sum(powers) / len(powers)
            total_avg_power += avg_device_power
        
        return total_avg_power
    
    def get_power_history(self, duration: timedelta = timedelta(hours=1)) -> List[PowerReading]:
        """Get power consumption history.
        
        Args:
            duration: Time period to retrieve
            
        Returns:
            List of power readings
        """
        cutoff_time = datetime.now() - duration
        return [
            r for r in self._readings 
            if r.timestamp > cutoff_time
        ]
    
    def get_power_stats(self) -> Dict[str, Any]:
        """Get power consumption statistics.
        
        Returns:
            Dictionary with power statistics
        """
        if not self._readings:
            return {"error": "No power readings available"}
        
        recent_readings = self.get_power_history(timedelta(hours=1))
        
        if not recent_readings:
            return {"error": "No recent power readings"}
        
        powers = [r.power_watts for r in recent_readings]
        
        return {
            "current_power_watts": self.get_current_power(),
            "avg_power_watts": sum(powers) / len(powers),
            "min_power_watts": min(powers),
            "max_power_watts": max(powers),
            "readings_count": len(recent_readings),
            "monitoring_duration_hours": 1.0,
            "devices": list(set(r.device for r in recent_readings))
        }


class PowerEstimator:
    """Estimate power consumption for training workloads."""
    
    def __init__(self, system_profile: Optional[SystemPowerProfile] = None):
        """Initialize power estimator.
        
        Args:
            system_profile: System power profile (uses defaults if None)
        """
        self.profile = system_profile or self._get_default_profile()
    
    def _get_default_profile(self) -> SystemPowerProfile:
        """Get default system power profile."""
        return SystemPowerProfile(
            idle_power_watts=50.0,          # Typical desktop idle power
            cpu_max_power_watts=65.0,       # Typical CPU TDP
            gpu_power_watts_per_device=250.0,  # Typical training GPU (RTX 3080/4080)
            memory_power_watts_per_gb=2.0,  # DDR4/5 power per GB
            network_power_watts=5.0,        # Network interface
            storage_power_watts=5.0         # SSD power
        )
    
    def estimate_training_power(
        self,
        num_gpus: int = 1,
        cpu_utilization: float = 0.8,
        memory_gb: int = 16,
        duration_hours: float = 1.0
    ) -> Dict[str, float]:
        """Estimate power consumption for training workload.
        
        Args:
            num_gpus: Number of GPUs used
            cpu_utilization: CPU utilization (0.0-1.0)  
            memory_gb: Memory usage in GB
            duration_hours: Training duration in hours
            
        Returns:
            Dictionary with power consumption breakdown
        """
        # Component power consumption
        idle_power = self.profile.idle_power_watts
        cpu_power = self.profile.cpu_max_power_watts * cpu_utilization
        gpu_power = self.profile.gpu_power_watts_per_device * num_gpus
        memory_power = self.profile.memory_power_watts_per_gb * memory_gb
        network_power = self.profile.network_power_watts
        storage_power = self.profile.storage_power_watts
        
        # Total power
        total_power_watts = (
            idle_power + cpu_power + gpu_power + 
            memory_power + network_power + storage_power
        )
        
        # Energy consumption
        total_energy_kwh = (total_power_watts * duration_hours) / 1000
        
        return {
            "idle_power_watts": idle_power,
            "cpu_power_watts": cpu_power,
            "gpu_power_watts": gpu_power,
            "memory_power_watts": memory_power,
            "network_power_watts": network_power,
            "storage_power_watts": storage_power,
            "total_power_watts": total_power_watts,
            "duration_hours": duration_hours,
            "total_energy_kwh": total_energy_kwh
        }
    
    def estimate_carbon_emissions(
        self,
        energy_kwh: float,
        carbon_intensity: float
    ) -> Dict[str, float]:
        """Estimate carbon emissions from energy consumption.
        
        Args:
            energy_kwh: Energy consumption in kWh
            carbon_intensity: Grid carbon intensity (gCO2/kWh)
            
        Returns:
            Dictionary with carbon emission estimates
        """
        emissions_g = energy_kwh * carbon_intensity
        emissions_kg = emissions_g / 1000
        
        # Equivalencies for context
        km_driven = emissions_kg * 2.31  # km driven in average car
        trees_needed = emissions_kg / 21.77  # trees needed to offset (annual absorption)
        
        return {
            "energy_kwh": energy_kwh,
            "carbon_intensity_g_per_kwh": carbon_intensity,
            "emissions_g_co2": emissions_g,
            "emissions_kg_co2": emissions_kg,
            "equivalent_km_driven": km_driven,
            "trees_needed_annual_offset": trees_needed
        }