#!/usr/bin/env python3
"""
Generation 1 Advanced Features Demo

Demonstrates the new advanced carbon-aware features:
- Multi-region orchestration
- Real-time optimization
- Federated carbon-aware learning
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta

from carbon_aware_trainer import (
    CarbonMonitor,
    MultiRegionOrchestrator,
    RealTimeOptimizer,
    CarbonAwareFederated,
    OptimizationMode,
    RegionConfig,
    FederatedClient
)
from carbon_aware_trainer.core.types import CarbonDataSource

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_multi_region_orchestration():
    """Demo multi-region training orchestration."""
    print("\n" + "="*60)
    print("🌍 Multi-Region Orchestration Demo")
    print("="*60)
    
    # Define regions with different characteristics (using available cached data)
    regions = {
        "US-CA": RegionConfig(
            region_id="US-CA",
            gpus=16,
            cost_per_hour=12.8,
            bandwidth_gbps=10,
            carbon_threshold=100
        ),
        "US-WA": RegionConfig(
            region_id="US-WA", 
            gpus=8,
            cost_per_hour=14.2,
            bandwidth_gbps=8,
            carbon_threshold=80
        )
    }
    
    # Create monitor for all regions using cached data for demo
    monitor = CarbonMonitor(
        regions=list(regions.keys()),
        data_source=CarbonDataSource.CACHED,
        api_key="sample_data/sample_carbon_data.json"
    )
    
    # Create orchestrator
    orchestrator = MultiRegionOrchestrator(
        regions=regions,
        monitor=monitor,
        migration_bandwidth_gbps=10.0,
        checkpoint_size_gb=5.0
    )
    
    # Optimize placement for a large training job
    print("🎯 Optimizing placement for multi-region training...")
    placement_plan = await orchestrator.optimize_placement(
        model_size_gb=50,
        dataset_size_gb=500,
        training_hours=168,  # 1 week
        carbon_budget_kg=1000,
        cost_budget_usd=5000
    )
    
    print(f"✅ Optimal placement plan:")
    print(f"   Primary region: {placement_plan.primary_region}")
    print(f"   Backup regions: {', '.join(placement_plan.backup_regions)}")
    print(f"   Expected carbon: {placement_plan.expected_carbon_kg:.2f} kg CO2")
    print(f"   Expected cost: ${placement_plan.expected_cost_usd:.2f}")
    print(f"   Carbon savings: {placement_plan.carbon_savings_pct:.1f}%")
    print(f"   Migration windows: {len(placement_plan.migration_windows)} identified")
    
    # Simulate training execution
    print("\n🚀 Executing multi-region training...")
    
    async def dummy_training_function():
        await asyncio.sleep(1)
        return {"loss": 0.5, "accuracy": 0.85}
    
    training_metrics = await orchestrator.execute_training(
        placement_plan=placement_plan,
        training_function=dummy_training_function,
        checkpoint_interval=timedelta(hours=1),
        migration_threshold=100
    )
    
    print(f"✅ Training completed:")
    print(f"   Duration: {training_metrics['duration_hours']:.2f} hours")
    print(f"   Regions used: {', '.join(training_metrics['regions_used'])}")
    print(f"   Migrations: {training_metrics['migrations']}")
    
    # Show migration history
    migration_history = orchestrator.get_migration_history()
    if migration_history:
        print(f"\n📊 Migration History:")
        for migration in migration_history:
            print(f"   {migration['timestamp'].strftime('%H:%M')} - {migration['from_region']} → {migration['to_region']}")


async def demo_real_time_optimization():
    """Demo real-time optimization engine."""
    print("\n" + "="*60)
    print("⚡ Real-Time Optimization Demo") 
    print("="*60)
    
    # Create optimizer in different modes
    modes = [OptimizationMode.CARBON_FIRST, OptimizationMode.BALANCED, OptimizationMode.PERFORMANCE_FIRST]
    
    for mode in modes:
        print(f"\n🧠 Testing {mode.value} mode...")
        
        # Create monitor for optimizer using cached data
        monitor = CarbonMonitor(
            regions=["US-CA"], 
            data_source=CarbonDataSource.CACHED,
            api_key="sample_data/sample_carbon_data.json"
        )
        
        optimizer = RealTimeOptimizer(
            monitor=monitor,
            mode=mode,
            optimization_interval=5  # 5 seconds for demo
        )
        
        # Callback to handle optimization actions
        async def training_callback(action):
            print(f"   📝 Action executed: {action.action_type}")
            print(f"      Rationale: {action.rationale}")
            print(f"      Expected carbon savings: {action.expected_carbon_savings:.1f} gCO2")
            print(f"      Confidence: {action.confidence:.2f}")
        
        # Start optimization
        await optimizer.start_optimization(
            region="US-CA",
            training_callback=training_callback
        )
        
        # Let it run for a bit
        await asyncio.sleep(12)
        
        # Stop optimization
        await optimizer.stop_optimization()
        
        # Show summary
        summary = optimizer.get_optimization_summary()
        print(f"   📊 Optimization Summary:")
        print(f"      Total optimizations: {summary.get('total_optimizations', 0)}")
        print(f"      Action breakdown: {summary.get('action_breakdown', {})}")
        print(f"      Estimated carbon savings: {summary.get('estimated_carbon_savings_kg', 0):.3f} kg")
        print(f"      Current state: {summary.get('current_state', 'unknown')}")


async def demo_federated_carbon_aware():
    """Demo federated learning with carbon awareness."""
    print("\n" + "="*60)
    print("🔗 Federated Carbon-Aware Learning Demo")
    print("="*60)
    
    # Define federated clients across different regions
    clients = {
        "edge-us-west": FederatedClient(
            client_id="edge-us-west",
            region="US-CA", 
            compute_capacity=0.8,
            data_samples=1000,
            carbon_threshold=100
        ),
        "edge-eu-central": FederatedClient(
            client_id="edge-eu-central",
            region="EU-DE",
            compute_capacity=0.6,
            data_samples=800,
            carbon_threshold=80
        ),
        "edge-asia-pacific": FederatedClient(
            client_id="edge-asia-pacific",
            region="AP-SG",
            compute_capacity=0.7,
            data_samples=1200,
            carbon_threshold=120
        ),
        "edge-us-east": FederatedClient(
            client_id="edge-us-east", 
            region="US-NY",
            compute_capacity=0.9,
            data_samples=900,
            carbon_threshold=90
        ),
        "edge-eu-west": FederatedClient(
            client_id="edge-eu-west",
            region="EU-FR", 
            compute_capacity=0.5,
            data_samples=700,
            carbon_threshold=85
        )
    }
    
    # Create monitor for federated learning using cached data
    all_regions = list(set(client.region for client in clients.values()))
    monitor = CarbonMonitor(
        regions=all_regions,
        data_source=CarbonDataSource.CACHED,
        api_key="sample_data/sample_carbon_data.json"
    )
    
    # Create federated orchestrator
    fed_orchestrator = CarbonAwareFederated(
        clients=clients,
        aggregation_server_region="US-CA",
        monitor=monitor,
        carbon_threshold=100
    )
    
    # Run multiple federated rounds
    print("🔄 Running carbon-aware federated learning rounds...")
    
    for round_num in range(3):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Select clients based on carbon intensity
        print("👥 Selecting clients based on carbon intensity...")
        selection = await fed_orchestrator.select_clients(
            round_num=round_num,
            num_clients=3,
            min_data_samples=500
        )
        
        print(f"✅ Selected clients: {', '.join(selection.selected_clients)}")
        print(f"   Expected total carbon: {selection.expected_carbon_total:.1f} gCO2/kWh")
        print(f"   Estimated training time: {selection.estimated_training_time:.1f} minutes")
        
        if selection.rejected_clients:
            print("❌ Rejected clients:")
            for client_id, reason in selection.rejected_clients.items():
                print(f"   {client_id}: {reason}")
        
        # Train on selected clients
        print("🏃‍♂️ Training on selected clients...")
        client_updates = []
        
        for client_id in selection.selected_clients:
            print(f"   Training on {client_id}...")
            update = await fed_orchestrator.train_client(
                client_id=client_id,
                model_parameters={"weights": f"model_round_{round_num}"},
                local_epochs=3,
                batch_size=32
            )
            
            if update:
                client_updates.append(update)
                print(f"   ✅ {client_id}: Avg carbon {update['average_carbon_intensity']:.1f} gCO2/kWh")
        
        # Carbon-aware aggregation
        print("🧮 Performing carbon-aware aggregation...")
        aggregation_result = await fed_orchestrator.carbon_aware_aggregate(
            client_updates=client_updates,
            max_wait_hours=2,
            min_updates=2
        )
        
        print(f"✅ Aggregation completed:")
        print(f"   Global model version: {aggregation_result['global_model_version']}")
        print(f"   Server carbon: {aggregation_result['server_carbon_intensity']:.1f} gCO2/kWh")
        print(f"   Average client carbon: {aggregation_result['average_client_carbon']:.1f} gCO2/kWh")
        savings = aggregation_result['carbon_savings_estimate']
        print(f"   Estimated carbon savings: {savings['total_estimated_savings_pct']:.1f}%")
    
    # Show federation summary
    print(f"\n📊 Federation Summary:")
    summary = fed_orchestrator.get_federation_summary()
    print(f"   Total rounds completed: {summary['total_rounds_completed']}")
    print(f"   Active clients: {summary['active_clients']}/{summary['total_clients']}")
    print(f"   Average client carbon: {summary['average_client_carbon']:.1f} gCO2/kWh")
    print(f"   Global model version: {summary['global_model_version']}")


async def main():
    """Run all Generation 1 demos."""
    print("🌱 Carbon-Aware-Trainer Generation 1 Advanced Features Demo")
    print("This demo showcases the enhanced capabilities for intelligent ML training.")
    
    try:
        # Run all demos
        await demo_multi_region_orchestration()
        await demo_real_time_optimization()
        await demo_federated_carbon_aware()
        
        print("\n" + "="*60)
        print("🎉 Generation 1 Demo Completed Successfully!")
        print("All advanced features are working and ready for production use.")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo encountered an error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    # Run the demo
    result = asyncio.run(main())
    exit(result)