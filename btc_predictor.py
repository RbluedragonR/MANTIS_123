#!/usr/bin/env python3

import os
import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

# Data sources
import requests
import yfinance as yf
try:
    import ccxt
except ImportError:
    ccxt = None
    print("[ERROR] ccxt is not installed. Some exchanges will not be available.")

# Technical indicators
try:
    import pandas_ta as ta
except ImportError:
    ta = None
    print("[ERROR] pandas_ta is not installed. Technical indicators will not be available.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline versioning
FEATURE_PIPELINE_VERSION = "1.0.0"

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")
logger.info(f"Feature pipeline version: {FEATURE_PIPELINE_VERSION}")

class BTCDataFetcher:
    """Fetches and processes Bitcoin market data from multiple sources."""
    def __init__(self):
        self.cache = None
        self.cache_time = None
        self.cache_hours = None
        if ccxt:
            self.exchanges = {
                'binance': ccxt.binance(),
                'coinbase': ccxt.coinbase(),  # Use latest supported Coinbase
                'kraken': ccxt.kraken(),
            }
        else:
            self.exchanges = {}

    def get_current_price(self) -> float:
        prices = []
        sources = [
            ("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", "binance"),
            ("https://api.coinbase.com/v2/prices/BTC-USD/spot", "coinbase"),
        ]
        for url, source in sources:
            try:
                response = requests.get(url, timeout=5)
                data = response.json()
                if source == "binance":
                    price = float(data["price"])
                elif source == "coinbase":
                    price = float(data["data"]["amount"])
                prices.append(price)
                logger.debug(f"Got price from {source}: {price}")
            except Exception as e:
                logger.warning(f"Failed to get price from {source}: {e}")
        if prices:
            return np.mean(prices)
        else:
            logger.warning("All price sources failed, returning 0.0")
            return 0.0

    def get_historical_data(self, hours: int = 168, cache: bool = True) -> pd.DataFrame:
        # Use cache if available and fresh
        if cache and self.cache is not None and self.cache_hours == hours and (time.time() - self.cache_time < 60):
            logger.info("Using cached historical data")
            return self.cache.copy()
        try:
            ticker = yf.Ticker("BTC-USD")
            data = ticker.history(period="7d", interval="1h")
            if len(data) < hours:
                data = ticker.history(period="30d", interval="1h")
            data = data.reset_index()
            data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            if len(data) < hours:
                logger.warning(f"Historical data too short: {len(data)} < {hours}")
            self.cache = data.tail(hours).copy()
            self.cache_time = time.time()
            self.cache_hours = hours
            return self.cache.copy()
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            # Fallback: return empty DataFrame
            return pd.DataFrame()

    def get_on_chain_data(self) -> Dict:
        try:
            return {
                'network_hashrate': 0.0,
                'difficulty': 0.0,
                'mempool_size': 0.0,
                'active_addresses': 0.0,
                'transaction_volume': 0.0,
            }
        except Exception as e:
            logger.warning(f"Failed to get on-chain data: {e}")
            return {}

class FeatureEngineering:
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_names = []

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if ta is None:
            logger.warning("pandas_ta not available, skipping technical indicators")
            return data
        if len(data) < 60:
            logger.warning(f"Not enough data for technical indicators: {len(data)} rows")
        # Price-based indicators
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        for period in [7, 14, 21, 50]:
            if len(data) >= period:
                data[f'sma_{period}'] = data['close'].rolling(period).mean()
                data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            else:
                data[f'sma_{period}'] = np.nan
                data[f'ema_{period}'] = np.nan
        if len(data) >= 21:
            data['rsi_14'] = ta.rsi(data['close'], length=14)
            data['rsi_21'] = ta.rsi(data['close'], length=21)
        else:
            data['rsi_14'] = np.nan
            data['rsi_21'] = np.nan
        if len(data) >= 26:
            macd = ta.macd(data['close'])
            data['macd'] = macd.get('MACD_12_26_9', np.nan)
            data['macd_signal'] = macd.get('MACDs_12_26_9', np.nan)
            data['macd_histogram'] = macd.get('MACDh_12_26_9', np.nan)
        else:
            data['macd'] = np.nan
            data['macd_signal'] = np.nan
            data['macd_histogram'] = np.nan
        if len(data) >= 20:
            bb = ta.bbands(data['close'], length=20)
            data['bb_upper'] = bb.get('BBU_20_2.0', np.nan)
            data['bb_middle'] = bb.get('BBM_20_2.0', np.nan)
            data['bb_lower'] = bb.get('BBL_20_2.0', np.nan)
            data['bb_width'] = (bb.get('BBU_20_2.0', 0) - bb.get('BBL_20_2.0', 0)) / (bb.get('BBM_20_2.0', 1) or 1)
        else:
            data['bb_upper'] = np.nan
            data['bb_middle'] = np.nan
            data['bb_lower'] = np.nan
            data['bb_width'] = np.nan
        data['volume_sma'] = data['volume'].rolling(20).mean() if len(data) >= 20 else np.nan
        data['volume_ratio'] = data['volume'] / data['volume_sma'] if len(data) >= 20 else np.nan
        data['volatility'] = data['returns'].rolling(24).std() if len(data) >= 24 else np.nan
        data['price_position'] = (data['close'] - data['low'].rolling(14).min()) / (data['high'].rolling(14).max() - data['low'].rolling(14).min()) if len(data) >= 14 else np.nan
        data['momentum'] = data['close'] / data['close'].shift(10) - 1 if len(data) >= 10 else np.nan
        return data

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['dow_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['dow_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        return data

    def add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        for lag in [1, 2, 3, 6, 12, 24]:
            data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
        for lag in [1, 6, 12, 24]:
            data[f'price_lag_{lag}'] = data['close'].shift(lag)
        return data

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = self.add_technical_indicators(df)
        data = self.add_time_features(data)
        data = self.add_lagged_features(data)
        data = data.dropna()
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        if len(feature_cols) < 40:
            logger.warning(f"Not enough features for embedding, padding with zeros. Got {len(feature_cols)} features.")
        return data

class BTCPredictionModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int] = [256, 128, 64], dropout: float = 0.3):
        super(BTCPredictionModel, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class BTCPredictor:
    def __init__(self, model_path: str = "btc_model.pth"):
        self.data_fetcher = BTCDataFetcher()
        self.feature_engineer = FeatureEngineering()
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.feature_names = []
        self.pipeline_version = FEATURE_PIPELINE_VERSION
        self.last_training_class_balance = None
        self.load_model()

    def collect_training_data(self, hours: int = 168) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Collecting training data...")
        df = self.data_fetcher.get_historical_data(hours)
        if df.empty or len(df) < 60:
            logger.warning(f"Insufficient historical data for training: {len(df)} rows")
            return np.zeros((1, 1)), np.zeros(1)
        df = self.feature_engineer.prepare_features(df)
        if len(df) == 0:
            logger.warning("No data after feature engineering. Returning zeros.")
            return np.zeros((1, 1)), np.zeros(1)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df[:-1]
        feature_cols = self.feature_engineer.feature_names
        X = df[feature_cols].values
        y = df['target'].values
        X = self.scaler.fit_transform(X)
        self.feature_names = feature_cols
        class_balance = np.bincount(y.astype(int))
        self.last_training_class_balance = class_balance
        logger.info(f"Training data shape: {X.shape}, Target distribution: {class_balance}")
        if len(class_balance) == 2 and abs(class_balance[0] - class_balance[1]) < 0.05 * sum(class_balance):
            logger.info("Class balance is close to 50/50 (as expected for BTC direction)")
        else:
            logger.warning(f"Class imbalance detected: {class_balance}")
        return X, y

    def train_model(self, retrain: bool = False):
        if self.model is not None and not retrain:
            logger.info("Model already loaded. Use retrain=True to retrain.")
            return
        logger.info(f"Training Bitcoin prediction model (pipeline version {self.pipeline_version})...")
        X, y = self.collect_training_data()
        if X.shape[0] < 2:
            logger.error("Not enough data to train model. Aborting training.")
            return
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        input_size = X_train.shape[1]
        self.model = BTCPredictionModel(input_size).to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(200):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                predictions = (val_outputs > 0.5).float()
                accuracy = (predictions == y_test_tensor).float().mean()
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
            if patience_counter >= 30:
                logger.info("Early stopping triggered")
                break
        logger.info("Model training completed")

    def predict_direction(self) -> float:
        if self.model is None:
            logger.error("Model not loaded. Please train the model first.")
            return 0.0
        try:
            df = self.data_fetcher.get_historical_data(hours=72)
            if df.empty or len(df) < 60:
                logger.warning("Insufficient data for prediction. Returning 0.5.")
                return 0.5
            df = self.feature_engineer.prepare_features(df)
            if len(df) == 0:
                logger.warning("No data after feature engineering. Returning 0.5.")
                return 0.5
            latest_features = df[self.feature_names].iloc[-1].values.reshape(1, -1)
            if latest_features.shape[1] != len(self.feature_names):
                logger.warning(f"Feature count mismatch: {latest_features.shape[1]} vs {len(self.feature_names)}")
            latest_features_scaled = self.scaler.transform(latest_features)
            self.model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(latest_features_scaled).to(DEVICE)
                prediction = self.model(input_tensor).cpu().numpy()[0][0]
            logger.info(f"Prediction probability: {prediction:.4f}")
            return prediction
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5

    def generate_mantis_embedding(self) -> List[float]:
        try:
            df = self.data_fetcher.get_historical_data(hours=72)
            if df.empty or len(df) < 60:
                logger.warning("Insufficient data for embedding. Using fallback random embedding.")
                return self._fallback_embedding(context="insufficient data")
            df = self.feature_engineer.prepare_features(df)
            if len(df) == 0:
                logger.warning("No data after feature engineering. Using fallback random embedding.")
                return self._fallback_embedding(context="no features")
            direction_prob = self.predict_direction()
            embedding = []
            prediction_signal = direction_prob
            for i in range(10):
                signal = prediction_signal + np.random.normal(0, 0.1) * (0.1 - 0.01 * i)
                embedding.append(np.clip(signal * 2 - 1, -1, 1))
            latest_features = df[self.feature_names].iloc[-1].values
            if len(latest_features) < 40:
                logger.warning(f"Not enough features for embedding, padding with zeros. Got {len(latest_features)} features.")
                latest_features = np.pad(latest_features, (0, 40 - len(latest_features)), 'constant')
            for i in range(40):
                normalized_val = np.tanh(latest_features[i])
                embedding.append(normalized_val)
            current_time = datetime.now()
            hour_signal = np.sin(2 * np.pi * current_time.hour / 24)
            dow_signal = np.sin(2 * np.pi * current_time.weekday() / 7)
            for i in range(20):
                time_signal = hour_signal if i % 2 == 0 else dow_signal
                embedding.append(time_signal + np.random.normal(0, 0.05))
            if len(df) > 24:
                volatility = df['close'].pct_change().rolling(24).std().iloc[-1]
                vol_signal = np.tanh(volatility * 100)
                for i in range(20):
                    embedding.append(vol_signal + np.random.normal(0, 0.1))
            else:
                embedding.extend([0.0] * 20)
            remaining = 100 - len(embedding)
            for i in range(remaining):
                noise = np.random.normal(prediction_signal - 0.5, 0.2)
                embedding.append(np.clip(noise, -1, 1))
            embedding = embedding[:100]
            embedding = [np.clip(val, -1, 1) for val in embedding]
            logger.info(f"Generated embedding with {len(embedding)} dimensions")
            logger.info(f"Embedding stats: min={min(embedding):.3f}, max={max(embedding):.3f}, mean={np.mean(embedding):.3f}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return self._fallback_embedding(context=str(e))

    def _fallback_embedding(self, context="unknown") -> List[float]:
        logger.warning(f"Using fallback random embedding due to: {context}")
        return [np.random.uniform(-1, 1) for _ in range(100)]

    def save_model(self):
        if self.model is not None:
            scaler_dict = {
                'mean_': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scale_': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'var_': self.scaler.var_.tolist() if hasattr(self.scaler, 'var_') else None,
                'n_features_in_': getattr(self.scaler, 'n_features_in_', None),
            }
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': scaler_dict,
                'feature_names': self.feature_names,
                'pipeline_version': self.pipeline_version
            }, self.model_path)
            logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=DEVICE)
                input_size = len(checkpoint['feature_names'])
                self.model = BTCPredictionModel(input_size).to(DEVICE)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                scaler_dict = checkpoint.get('scaler', {})
                scaler = StandardScaler()
                if scaler_dict.get('mean_') is not None:
                    scaler.mean_ = np.array(scaler_dict['mean_'])
                if scaler_dict.get('scale_') is not None:
                    scaler.scale_ = np.array(scaler_dict['scale_'])
                if scaler_dict.get('var_') is not None:
                    scaler.var_ = np.array(scaler_dict['var_'])
                if scaler_dict.get('n_features_in_') is not None:
                    scaler.n_features_in_ = scaler_dict['n_features_in_']
                self.scaler = scaler
                self.feature_names = checkpoint['feature_names']
                self.pipeline_version = checkpoint.get('pipeline_version', 'unknown')
                logger.info(f"Model loaded from {self.model_path} (pipeline version {self.pipeline_version})")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            self.model = None

def main():
    logger.info("Initializing BTC Predictor...")
    logger.info(f"Feature pipeline version: {FEATURE_PIPELINE_VERSION}")
    predictor = BTCPredictor()
    if predictor.model is None:
        logger.info("Training new model...")
        predictor.train_model()
    logger.info("Testing prediction...")
    embedding = predictor.generate_mantis_embedding()
    logger.info(f"Generated embedding for MANTIS: {len(embedding)} dimensions")
    logger.info(f"Sample values: {embedding[:10]}")
    return embedding

if __name__ == "__main__":
    main() 