"""Command-line interface for carbon-aware trainer."""

import asyncio
import click
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .core.monitor import CarbonMonitor
from .core.scheduler import CarbonAwareTrainer
from .core.forecasting import CarbonForecaster
from .core.types import CarbonDataSource, TrainingConfig
from .strategies.threshold import ThresholdScheduler
from .strategies.adaptive import AdaptiveScheduler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Carbon-Aware-Trainer: Intelligent ML training scheduler for carbon reduction."""
    pass


@cli.command()
@click.option('--region', '-r', default='US-CA', help='Training region code')
@click.option('--data-source', '-d', default='electricitymap', 
              type=click.Choice(['electricitymap', 'watttime', 'cached']),
              help='Carbon data source')
@click.option('--api-key', '-k', help='API key for data source')
@click.option('--duration', '-t', default=24, help='Monitoring duration in hours')
@click.option('--output', '-o', help='Output file for monitoring data')
def monitor(region: str, data_source: str, api_key: Optional[str], duration: int, output: Optional[str]):
    """Monitor carbon intensity for a region."""
    
    async def run_monitor():
        # Convert string to enum
        data_source_enum = getattr(CarbonDataSource, data_source.upper())
        
        monitor = CarbonMonitor(
            regions=[region],
            data_source=data_source_enum,
            api_key=api_key,
            update_interval=300  # 5 minutes
        )
        
        monitoring_data = []
        
        def save_data(event_type, data):
            if event_type == 'intensity_change':
                monitoring_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'region': data['region'],
                    'carbon_intensity': data['new_intensity'].carbon_intensity,
                    'renewable_percentage': data['new_intensity'].renewable_percentage
                })
        
        async with monitor:
            monitor.add_callback(save_data)
            await monitor.start_monitoring()
            
            click.echo(f"Monitoring carbon intensity for {region} using {data_source}")
            click.echo(f"Duration: {duration} hours")
            click.echo("Press Ctrl+C to stop")
            
            try:
                await asyncio.sleep(duration * 3600)
            except KeyboardInterrupt:
                click.echo("\nStopping monitor...")
            
            await monitor.stop_monitoring()
        
        # Save data if output specified
        if output and monitoring_data:
            with open(output, 'w') as f:
                json.dump(monitoring_data, f, indent=2)
            click.echo(f"Saved {len(monitoring_data)} data points to {output}")
        
        # Show summary
        if monitoring_data:
            intensities = [d['carbon_intensity'] for d in monitoring_data]
            click.echo(f"\nSummary for {region}:")
            click.echo(f"  Data points: {len(monitoring_data)}")
            click.echo(f"  Avg carbon intensity: {sum(intensities)/len(intensities):.1f} gCO2/kWh")
            click.echo(f"  Min carbon intensity: {min(intensities):.1f} gCO2/kWh")
            click.echo(f"  Max carbon intensity: {max(intensities):.1f} gCO2/kWh")
    
    asyncio.run(run_monitor())


@cli.command()
@click.option('--region', '-r', default='US-CA', help='Training region code')
@click.option('--data-source', '-d', default='electricitymap', 
              type=click.Choice(['electricitymap', 'watttime', 'cached']),
              help='Carbon data source')
@click.option('--api-key', '-k', help='API key for data source')
@click.option('--hours', '-h', default=24, help='Forecast duration in hours')
@click.option('--threshold', '-t', default=100.0, help='Carbon threshold (gCO2/kWh)')
@click.option('--duration', default=8, help='Training duration in hours')
def forecast(region: str, data_source: str, api_key: Optional[str], 
            hours: int, threshold: float, duration: int):
    """Get carbon intensity forecast and find optimal training windows."""
    
    async def run_forecast():
        data_source_enum = getattr(CarbonDataSource, data_source.upper())
        
        monitor = CarbonMonitor(
            regions=[region],
            data_source=data_source_enum,
            api_key=api_key
        )
        
        async with monitor:
            # Get current intensity
            current = await monitor.get_current_intensity(region)
            if current:
                click.echo(f"Current carbon intensity in {region}: {current.carbon_intensity:.1f} gCO2/kWh")
                if current.renewable_percentage:
                    click.echo(f"Renewable percentage: {current.renewable_percentage:.1f}%")
            
            # Get forecast
            forecast_data = await monitor.get_forecast(region, hours=hours)
            if forecast_data and forecast_data.data_points:
                click.echo(f"\nForecast for next {hours} hours:")
                
                intensities = [p.carbon_intensity for p in forecast_data.data_points]
                click.echo(f"  Avg intensity: {sum(intensities)/len(intensities):.1f} gCO2/kWh")
                click.echo(f"  Min intensity: {min(intensities):.1f} gCO2/kWh")
                click.echo(f"  Max intensity: {max(intensities):.1f} gCO2/kWh")
                
                # Find optimal window
                forecaster = CarbonForecaster(monitor)
                windows = await forecaster.find_optimal_windows(
                    region=region,
                    duration_hours=duration,
                    num_windows=3,
                    max_carbon_intensity=threshold
                )
                
                if windows:
                    click.echo(f"\nOptimal {duration}h training windows (threshold: {threshold} gCO2/kWh):")
                    for i, window in enumerate(windows[:3], 1):
                        delay_hours = (window.start_time - datetime.now()).total_seconds() / 3600
                        click.echo(f"  {i}. Start: {window.start_time.strftime('%Y-%m-%d %H:%M')}")
                        click.echo(f"     Avg intensity: {window.avg_carbon_intensity:.1f} gCO2/kWh")
                        click.echo(f"     Renewable: {window.renewable_percentage:.1f}%")
                        click.echo(f"     Delay: {delay_hours:.1f} hours")
                        click.echo(f"     Confidence: {window.confidence_score:.2f}")
                        click.echo()
                else:
                    click.echo(f"\nNo suitable {duration}h windows found within threshold {threshold} gCO2/kWh")
            else:
                click.echo("No forecast data available")
    
    asyncio.run(run_forecast())


@cli.command()
@click.option('--region', '-r', default='US-CA', help='Training region code')
@click.option('--threshold', '-t', default=100.0, help='Carbon threshold (gCO2/kWh)')
@click.option('--data-source', '-d', default='electricitymap', 
              type=click.Choice(['electricitymap', 'watttime', 'cached']),
              help='Carbon data source')
@click.option('--api-key', '-k', help='API key for data source')
@click.option('--check-interval', default=300, help='Check interval in seconds')
@click.option('--max-duration', default=24, help='Max simulation duration in hours')
def simulate(region: str, threshold: float, data_source: str, api_key: Optional[str],
            check_interval: int, max_duration: int):
    """Simulate carbon-aware training with real-time monitoring."""
    
    async def run_simulation():
        click.echo(f"Simulating carbon-aware training in {region}")
        click.echo(f"Threshold: {threshold} gCO2/kWh")
        click.echo(f"Check interval: {check_interval} seconds")
        click.echo("Press Ctrl+C to stop\n")
        
        config = TrainingConfig(
            carbon_threshold=threshold,
            pause_threshold=threshold * 1.5,
            resume_threshold=threshold * 0.8,
            check_interval=check_interval
        )
        
        trainer = CarbonAwareTrainer(
            carbon_model=data_source,
            region=region,
            config=config,
            api_key=api_key
        )
        
        def on_state_change(state, metrics):
            click.echo(f"[{datetime.now().strftime('%H:%M:%S')}] State: {state.value}")
            current_metrics = trainer.get_carbon_metrics()
            click.echo(f"  Total carbon: {current_metrics['total_carbon_kg']:.3f} kg CO2")
            click.echo(f"  Runtime: {current_metrics['runtime_hours']:.2f} hours")
            if current_metrics['current_intensity']:
                click.echo(f"  Current intensity: {current_metrics['current_intensity']:.1f} gCO2/kWh")
        
        try:
            async with trainer:
                trainer.add_state_callback(on_state_change)
                
                async with trainer.training_session():
                    # Simulate training steps
                    simulation_end = datetime.now() + timedelta(hours=max_duration)
                    step = 0
                    
                    while datetime.now() < simulation_end:
                        # Simulate training step
                        step += 1
                        trainer.step = step
                        
                        # Wait between steps (simulate training time)
                        await asyncio.sleep(10)  # 10 seconds per "step"
                        
                        if step % 10 == 0:  # Every 100 seconds
                            metrics = trainer.get_carbon_metrics()
                            click.echo(f"Step {step}: {metrics['total_carbon_kg']:.3f} kg CO2 total")
        
        except KeyboardInterrupt:
            click.echo("\nStopping simulation...")
        
        # Final summary
        final_metrics = trainer.get_carbon_metrics()
        click.echo(f"\nSimulation Summary:")
        click.echo(f"  Total steps: {final_metrics['step']}")
        click.echo(f"  Total runtime: {final_metrics['runtime_hours']:.2f} hours")
        click.echo(f"  Total carbon: {final_metrics['total_carbon_kg']:.3f} kg CO2")
        click.echo(f"  Avg carbon intensity: {final_metrics['avg_carbon_intensity']:.1f} gCO2/kWh")
        click.echo(f"  Paused duration: {final_metrics['paused_duration_hours']:.2f} hours")
    
    asyncio.run(run_simulation())


@cli.command()
@click.option('--regions', '-r', default='US-CA,US-WA,EU-FR', help='Comma-separated region codes')
@click.option('--data-source', '-d', default='electricitymap', 
              type=click.Choice(['electricitymap', 'watttime', 'cached']),
              help='Carbon data source')
@click.option('--api-key', '-k', help='API key for data source')
def compare_regions(regions: str, data_source: str, api_key: Optional[str]):
    """Compare current carbon intensity across multiple regions."""
    
    async def run_comparison():
        region_list = [r.strip() for r in regions.split(',')]
        data_source_enum = getattr(CarbonDataSource, data_source.upper())
        
        monitor = CarbonMonitor(
            regions=region_list,
            data_source=data_source_enum,
            api_key=api_key
        )
        
        async with monitor:
            click.echo("Comparing carbon intensity across regions:\n")
            
            intensities = {}
            for region in region_list:
                intensity = await monitor.get_current_intensity(region)
                if intensity:
                    intensities[region] = intensity
                    
            if intensities:
                # Sort by carbon intensity
                sorted_regions = sorted(
                    intensities.items(), 
                    key=lambda x: x[1].carbon_intensity
                )
                
                click.echo("Regions ranked by carbon intensity (cleanest first):")
                for i, (region, intensity) in enumerate(sorted_regions, 1):
                    status = "🟢" if intensity.carbon_intensity < 100 else "🟡" if intensity.carbon_intensity < 200 else "🔴"
                    click.echo(f"{i}. {region}: {intensity.carbon_intensity:.1f} gCO2/kWh {status}")
                    if intensity.renewable_percentage:
                        click.echo(f"   Renewable: {intensity.renewable_percentage:.1f}%")
                
                # Recommend cleanest region
                cleanest_region = sorted_regions[0][0]
                cleanest_intensity = sorted_regions[0][1].carbon_intensity
                click.echo(f"\n🌱 Recommended region: {cleanest_region} ({cleanest_intensity:.1f} gCO2/kWh)")
            else:
                click.echo("No data available for any region")
    
    asyncio.run(run_comparison())


@cli.command()
@click.option('--output-dir', '-o', default='./sample_data', help='Output directory')
@click.option('--regions', '-r', default='US-CA,US-WA,EU-FR,EU-DE', help='Comma-separated regions')
@click.option('--format', '-f', default='json', type=click.Choice(['json', 'csv']),
              help='Output format')
def generate_sample_data(output_dir: str, regions: str, format: str):
    """Generate sample carbon intensity data for testing."""
    from .carbon_models.cached import CachedProvider
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    region_list = [r.strip() for r in regions.split(',')]
    
    if format == 'json':
        sample_file = output_path / 'sample_carbon_data.json'
        CachedProvider.create_sample_data(sample_file, region_list)
        click.echo(f"Generated sample data for {len(region_list)} regions: {sample_file}")
        click.echo(f"Use with: carbon-trainer monitor --data-source cached --api-key {sample_file}")
    else:
        click.echo("CSV format not yet supported for sample data generation")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()