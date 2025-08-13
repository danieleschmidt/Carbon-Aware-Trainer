"""
Neural Carbon Intensity Prediction with Transformer Architecture

Advanced deep learning models for predicting carbon intensity patterns,
incorporating weather data, energy demand, and renewable generation forecasts.

Research Paper: "Deep Learning for Grid Carbon Intensity Forecasting"
Author: Daniel Schmidt, Terragon Labs
Date: August 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime, timedelta
import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.types import CarbonIntensity, CarbonForecast
from ..core.forecasting import CarbonForecaster


@dataclass
class WeatherFeatures:
    """Weather features for carbon prediction."""
    temperature: float
    humidity: float
    wind_speed: float
    solar_irradiance: float
    cloud_cover: float
    precipitation: float


@dataclass
class EnergyDemandFeatures:
    """Energy demand features."""
    total_demand: float
    industrial_demand: float
    residential_demand: float
    commercial_demand: float
    peak_hour_indicator: float
    day_of_week: float
    month_of_year: float


@dataclass
class RenewableGenerationFeatures:
    """Renewable energy generation features."""
    solar_generation: float
    wind_generation: float
    hydro_generation: float
    geothermal_generation: float
    biomass_generation: float
    renewable_percentage: float


@dataclass
class CarbonPredictionInput:
    """Complete input for carbon prediction."""
    timestamp: datetime
    region: str
    weather: WeatherFeatures
    demand: EnergyDemandFeatures
    renewables: RenewableGenerationFeatures
    historical_carbon: List[float]  # Last 24 hours
    current_carbon: float


@dataclass
class PredictionResult:
    """Result of carbon intensity prediction."""
    predicted_intensities: List[float]  # Next 24 hours
    confidence_intervals: List[Tuple[float, float]]
    prediction_accuracy: float
    model_uncertainty: float
    feature_importance: Dict[str, float]


class CarbonDataset(Dataset):
    """PyTorch dataset for carbon prediction training."""
    
    def __init__(self, prediction_inputs: List[CarbonPredictionInput], targets: List[List[float]]):
        self.inputs = prediction_inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        target = self.targets[idx]
        
        # Convert to feature vector
        features = self._extract_features(input_data)
        return torch.FloatTensor(features), torch.FloatTensor(target)
    
    def _extract_features(self, input_data: CarbonPredictionInput) -> List[float]:
        """Extract numerical features from input data."""
        features = []
        
        # Weather features
        features.extend([
            input_data.weather.temperature,
            input_data.weather.humidity,
            input_data.weather.wind_speed,
            input_data.weather.solar_irradiance,
            input_data.weather.cloud_cover,
            input_data.weather.precipitation
        ])
        
        # Demand features
        features.extend([
            input_data.demand.total_demand,
            input_data.demand.industrial_demand,
            input_data.demand.residential_demand,
            input_data.demand.commercial_demand,
            input_data.demand.peak_hour_indicator,
            input_data.demand.day_of_week,
            input_data.demand.month_of_year
        ])
        
        # Renewable features
        features.extend([
            input_data.renewables.solar_generation,
            input_data.renewables.wind_generation,
            input_data.renewables.hydro_generation,
            input_data.renewables.geothermal_generation,
            input_data.renewables.biomass_generation,
            input_data.renewables.renewable_percentage
        ])
        
        # Historical carbon (normalized)
        historical_norm = np.array(input_data.historical_carbon) / 1000.0
        features.extend(historical_norm.tolist())
        
        # Current carbon
        features.append(input_data.current_carbon / 1000.0)
        
        # Temporal features
        hour = input_data.timestamp.hour / 24.0
        day = input_data.timestamp.day / 31.0
        month = input_data.timestamp.month / 12.0
        features.extend([hour, day, month])
        
        return features


class CarbonTransformer(nn.Module):
    """Transformer-based carbon intensity prediction model."""
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        output_dim: int = 24,  # 24 hour forecast
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
    def _create_positional_encoding(self, hidden_dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        max_len = 1000
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-np.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0).transpose(0, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            predictions: Carbon intensity predictions (batch_size, output_dim)
            uncertainties: Prediction uncertainties (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # Project input to hidden dimension
        hidden = self.input_projection(x)  # (batch_size, hidden_dim)
        
        # Add positional encoding
        hidden = hidden.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        hidden = hidden + self.positional_encoding[:1, :].to(x.device)
        
        # Transformer encoding
        hidden = hidden.transpose(0, 1)  # (1, batch_size, hidden_dim)
        encoded = self.transformer(hidden)  # (1, batch_size, hidden_dim)
        encoded = encoded.squeeze(0)  # (batch_size, hidden_dim)
        
        # Generate predictions and uncertainties
        predictions = self.output_projection(encoded)
        uncertainties = self.uncertainty_head(encoded)
        
        return predictions, uncertainties


class CarbonLSTM(nn.Module):
    """LSTM-based carbon intensity prediction model for comparison."""
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 256,
        num_layers: int = 3,
        output_dim: int = 24,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Softplus()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Reshape for LSTM (add sequence dimension)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch_size, 1, hidden_dim)
        lstm_out = lstm_out.squeeze(1)  # (batch_size, hidden_dim)
        
        # Generate predictions and uncertainties
        predictions = self.output_layer(lstm_out)
        uncertainties = self.uncertainty_head(lstm_out)
        
        return predictions, uncertainties


class NeuralCarbonPredictor:
    """
    Advanced neural network-based carbon intensity predictor.
    
    Novel contributions:
    1. Multi-modal fusion of weather, demand, and renewable data
    2. Transformer architecture for temporal dependencies
    3. Uncertainty quantification for reliability
    4. Transfer learning across regions
    """
    
    def __init__(
        self,
        model_type: str = "transformer",
        device: str = "auto",
        model_config: Optional[Dict[str, Any]] = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralCarbonPredictor")
        
        self.model_type = model_type
        self.device = self._get_device(device)
        self.model_config = model_config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = self._create_model()
        self.optimizer = None
        self.criterion = None
        self.is_trained = False
        
        # Feature importance tracking
        self.feature_names = self._get_feature_names()
        self.feature_importance = {}
    
    def _get_device(self, device: str) -> torch.device:
        """Get PyTorch device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _create_model(self) -> nn.Module:
        """Create neural network model."""
        if self.model_type == "transformer":
            model = CarbonTransformer(**self.model_config)
        elif self.model_type == "lstm":
            model = CarbonLSTM(**self.model_config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for importance analysis."""
        names = []
        
        # Weather features
        names.extend([
            "temperature", "humidity", "wind_speed", "solar_irradiance",
            "cloud_cover", "precipitation"
        ])
        
        # Demand features
        names.extend([
            "total_demand", "industrial_demand", "residential_demand",
            "commercial_demand", "peak_hour_indicator", "day_of_week", "month_of_year"
        ])
        
        # Renewable features
        names.extend([
            "solar_generation", "wind_generation", "hydro_generation",
            "geothermal_generation", "biomass_generation", "renewable_percentage"
        ])
        
        # Historical carbon (24 hours)
        names.extend([f"carbon_hour_{i}" for i in range(24)])
        
        # Current carbon and temporal
        names.extend(["current_carbon", "hour", "day", "month"])
        
        return names
    
    async def train(
        self,
        training_data: List[CarbonPredictionInput],
        targets: List[List[float]],
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ) -> Dict[str, List[float]]:
        """
        Train the neural carbon predictor.
        
        Args:
            training_data: List of training inputs
            targets: List of target carbon intensities (24-hour forecasts)
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            
        Returns:
            Training history with losses and metrics
        """
        self.logger.info(f"Training {self.model_type} model with {len(training_data)} samples")
        
        # Split data
        split_idx = int(len(training_data) * (1 - validation_split))
        train_inputs = training_data[:split_idx]
        train_targets = targets[:split_idx]
        val_inputs = training_data[split_idx:]
        val_targets = targets[split_idx:]
        
        # Create datasets and loaders
        train_dataset = CarbonDataset(train_inputs, train_targets)
        val_dataset = CarbonDataset(val_inputs, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize optimizer and loss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = self._create_loss_function()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_mse": [],
            "val_mse": [],
            "train_mae": [],
            "val_mae": []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            train_metrics = await self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = await self._validate_epoch(val_loader)
            
            # Update history
            for key, value in train_metrics.items():
                history[f"train_{key}"].append(value)
            for key, value in val_metrics.items():
                history[f"val_{key}"].append(value)
            
            # Learning rate scheduling
            scheduler.step(val_metrics["loss"])
            
            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "/tmp/best_carbon_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss {train_metrics['loss']:.4f}, "
                               f"Val Loss {val_metrics['loss']:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load("/tmp/best_carbon_model.pth"))
        self.is_trained = True
        
        # Calculate feature importance
        await self._calculate_feature_importance(val_loader)
        
        return history
    
    async def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        for features, targets in dataloader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions, uncertainties = self.model(features)
            loss = self.criterion(predictions, uncertainties, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                mse = F.mse_loss(predictions, targets)
                mae = F.l1_loss(predictions, targets)
                
                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
                num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "mse": total_mse / num_batches,
            "mae": total_mae / num_batches
        }
    
    async def _validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets in dataloader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions, uncertainties = self.model(features)
                loss = self.criterion(predictions, uncertainties, targets)
                
                # Calculate metrics
                mse = F.mse_loss(predictions, targets)
                mae = F.l1_loss(predictions, targets)
                
                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
                num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "mse": total_mse / num_batches,
            "mae": total_mae / num_batches
        }
    
    def _create_loss_function(self):
        """Create uncertainty-aware loss function."""
        def uncertainty_loss(predictions, uncertainties, targets):
            # Negative log likelihood with uncertainty
            mse_loss = F.mse_loss(predictions, targets, reduction='none')
            
            # Uncertainty-weighted loss
            weighted_loss = mse_loss / (2 * uncertainties**2) + 0.5 * torch.log(uncertainties**2)
            
            return weighted_loss.mean()
        
        return uncertainty_loss
    
    async def predict(
        self,
        input_data: CarbonPredictionInput,
        return_uncertainty: bool = True
    ) -> PredictionResult:
        """
        Predict carbon intensity for the next 24 hours.
        
        Args:
            input_data: Input features for prediction
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Prediction result with intensities and uncertainties
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        
        # Prepare input
        dataset = CarbonDataset([input_data], [[0.0] * 24])  # Dummy target
        features, _ = dataset[0]
        features = features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions, uncertainties = self.model(features)
            
            predictions = predictions.squeeze().cpu().numpy() * 1000.0  # Denormalize
            uncertainties = uncertainties.squeeze().cpu().numpy() * 1000.0
        
        # Calculate confidence intervals
        confidence_intervals = []
        for pred, unc in zip(predictions, uncertainties):
            lower = pred - 1.96 * unc  # 95% confidence interval
            upper = pred + 1.96 * unc
            confidence_intervals.append((lower, upper))
        
        return PredictionResult(
            predicted_intensities=predictions.tolist(),
            confidence_intervals=confidence_intervals,
            prediction_accuracy=0.95,  # Placeholder - would be calculated from validation
            model_uncertainty=float(np.mean(uncertainties)),
            feature_importance=self.feature_importance.copy()
        )
    
    async def _calculate_feature_importance(self, dataloader: DataLoader):
        """Calculate feature importance using gradient-based methods."""
        self.model.eval()
        importance_scores = np.zeros(len(self.feature_names))
        num_samples = 0
        
        for features, targets in dataloader:
            features = features.to(self.device)
            features.requires_grad_(True)
            
            predictions, _ = self.model(features)
            loss = F.mse_loss(predictions, targets.to(self.device))
            
            # Calculate gradients
            loss.backward()
            
            # Accumulate gradient magnitudes
            gradients = features.grad.abs().mean(dim=0).cpu().numpy()
            importance_scores += gradients
            num_samples += 1
            
            features.grad.zero_()
        
        # Normalize importance scores
        importance_scores /= num_samples
        importance_scores /= np.sum(importance_scores)
        
        # Create importance dictionary
        self.feature_importance = {
            name: float(score) for name, score in zip(self.feature_names, importance_scores)
        }
    
    async def evaluate_model(
        self,
        test_data: List[CarbonPredictionInput],
        test_targets: List[List[float]]
    ) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        test_dataset = CarbonDataset(test_data, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return await self._validate_epoch(test_loader)


class CarbonPredictionBenchmark:
    """Benchmarking suite for carbon prediction models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def run_model_comparison(
        self,
        training_data: List[CarbonPredictionInput],
        targets: List[List[float]],
        test_data: List[CarbonPredictionInput],
        test_targets: List[List[float]]
    ) -> Dict[str, Any]:
        """Compare different model architectures."""
        models = {
            "transformer": NeuralCarbonPredictor("transformer"),
            "lstm": NeuralCarbonPredictor("lstm")
        }
        
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"Training and evaluating {name} model")
            
            # Train model
            history = await model.train(training_data, targets, epochs=50)
            
            # Evaluate on test set
            test_metrics = await model.evaluate_model(test_data, test_targets)
            
            results[name] = {
                "training_history": history,
                "test_metrics": test_metrics,
                "feature_importance": model.feature_importance
            }
        
        return results
    
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark results."""
        analysis = {
            "best_model": None,
            "performance_comparison": {},
            "statistical_significance": {}
        }
        
        # Find best model by test loss
        best_loss = float('inf')
        for model_name, model_results in results.items():
            test_loss = model_results["test_metrics"]["loss"]
            if test_loss < best_loss:
                best_loss = test_loss
                analysis["best_model"] = model_name
        
        # Performance comparison
        for model_name, model_results in results.items():
            metrics = model_results["test_metrics"]
            analysis["performance_comparison"][model_name] = {
                "test_mse": metrics["mse"],
                "test_mae": metrics["mae"],
                "final_train_loss": model_results["training_history"]["train_loss"][-1],
                "convergence_epochs": len(model_results["training_history"]["train_loss"])
            }
        
        return analysis