import pytest
import numpy as np
import pandas as pd
from src.volatility.vol_forecasting import VolatilityForecaster

@pytest.fixture
def dummy_market_data():
    # Construct 100 days of dummy market data (Close and Volume)
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
    
    # Random walk close prices
    returns = np.random.normal(0.0005, 0.02, 100)
    prices = 100.0 * np.exp(np.cumsum(returns))
    
    # Volumes
    volumes = np.random.randint(10000, 50000, 100).astype(float)
    
    return pd.DataFrame({
        "Close": prices,
        "Volume": volumes
    }, index=dates)

def test_vol_forecasting_feature_engineering(dummy_market_data):
    # Add High and Low columns to dummy data for range-based Parkinson estimator
    dummy_market_data["High"] = dummy_market_data["Close"] * 1.01
    dummy_market_data["Low"] = dummy_market_data["Close"] * 0.99
    
    forecaster = VolatilityForecaster(dummy_market_data)
    X, y = forecaster.engineer_features(target_horizon=5)
    
    # We should have features and a target
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    
    expected_cols = [
        "returns", "vol_1", "vol_5", "vol_10", "vol_21",
        "vol_parkinson", "skew_21", "kurt_21"
    ]
    assert list(X.columns) == expected_cols
    # No NaNs should remain after dropping
    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0
    assert len(X) == len(y)
    assert len(X) > 0

def test_vol_forecasting_training_and_evaluation(dummy_market_data):
    # Add High and Low columns to dummy data
    dummy_market_data["High"] = dummy_market_data["Close"] * 1.01
    dummy_market_data["Low"] = dummy_market_data["Close"] * 0.99
    
    forecaster = VolatilityForecaster(dummy_market_data)
    X, y = forecaster.engineer_features(target_horizon=5)
    
    results = forecaster.train_and_evaluate(X, y, train_size=0.70)
    
    expected_models = [
        "Historical Baseline",
        "EWMA Volatility",
        "GARCH(1,1)",
        "HAR-RV Model",
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ]
    
    for model_name in expected_models:
        assert model_name in results
        metrics = results[model_name]
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert isinstance(metrics["predictions"], pd.Series)
        assert len(metrics["predictions"]) > 0

def test_vol_forecasting_input_validation():
    with pytest.raises(TypeError):
        VolatilityForecaster([1, 2, 3]) # type: ignore
        
    df_missing_cols = pd.DataFrame({"Close": [100.0] * 70})
    with pytest.raises(ValueError):
        VolatilityForecaster(df_missing_cols)
        
    df_too_short = pd.DataFrame({"Close": [100.0] * 10, "Volume": [1000] * 10})
    with pytest.raises(ValueError):
        VolatilityForecaster(df_too_short)
