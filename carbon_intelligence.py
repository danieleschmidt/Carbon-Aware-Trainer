
#!/usr/bin/env python3
"""
Carbon Intelligence Service - Real-time optimization.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

class CarbonIntelligenceService:
    def __init__(self):
        self.regions = [
            "US-CA", "US-WA", "US-TX", "EU-FR", 
            "EU-DE", "EU-NO", "AP-SE", "AP-NE"
        ]
        self.thresholds = {
            "low": 50,      # gCO2/kWh
            "medium": 100,
            "high": 150,
            "critical": 200
        }
    
    async def get_global_carbon_map(self) -> Dict[str, Any]:
        """Get current carbon intensity for all regions."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for region in self.regions:
                task = self.get_region_intensity(session, region)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            carbon_map = {}
            for region, result in zip(self.regions, results):
                if isinstance(result, Exception):
                    carbon_map[region] = {"intensity": 150.0, "status": "error"}
                else:
                    carbon_map[region] = result
            
            return carbon_map
    
    async def get_region_intensity(self, session, region: str) -> Dict[str, Any]:
        """Get carbon intensity for a specific region."""
        # Mock implementation - in production, use real APIs
        mock_intensities = {
            "US-CA": 85.0, "US-WA": 45.0, "US-TX": 120.0,
            "EU-FR": 65.0, "EU-DE": 95.0, "EU-NO": 25.0,
            "AP-SE": 110.0, "AP-NE": 130.0
        }
        
        intensity = mock_intensities.get(region, 100.0)
        status = self.classify_intensity(intensity)
        
        return {
            "intensity": intensity,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "renewable_percentage": max(0, 100 - intensity)
        }
    
    def classify_intensity(self, intensity: float) -> str:
        """Classify carbon intensity level."""
        if intensity <= self.thresholds["low"]:
            return "low"
        elif intensity <= self.thresholds["medium"]:
            return "medium"
        elif intensity <= self.thresholds["high"]:
            return "high"
        else:
            return "critical"
    
    async def optimize_placement(self, requirements: Dict) -> Dict[str, Any]:
        """Optimize workload placement based on carbon intensity."""
        carbon_map = await self.get_global_carbon_map()
        
        # Sort regions by carbon intensity
        sorted_regions = sorted(
            carbon_map.items(),
            key=lambda x: x[1]["intensity"]
        )
        
        optimal_regions = []
        for region, data in sorted_regions[:3]:  # Top 3 cleanest
            if data["intensity"] <= requirements.get("max_carbon", 150):
                optimal_regions.append({
                    "region": region,
                    "carbon_intensity": data["intensity"],
                    "status": data["status"],
                    "renewable_percentage": data.get("renewable_percentage", 50)
                })
        
        return {
            "recommendation": optimal_regions[0] if optimal_regions else None,
            "alternatives": optimal_regions[1:3],
            "carbon_map": carbon_map,
            "optimization_timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    service = CarbonIntelligenceService()
    
    async def main():
        result = await service.optimize_placement({"max_carbon": 100})
        print(json.dumps(result, indent=2))
    
    asyncio.run(main())
