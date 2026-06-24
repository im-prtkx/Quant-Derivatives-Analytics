import math
import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from arch import arch_model

logger = logging.getLogger(__name__)

class VolatilityForecaster:
    """
    Advanced forecasting framework for predicting future realized volatility.
    Implements Historical Baseline, EWMA, GARCH(1,1), HAR-RV, and Machine Learning models
    using a rolling walk-forward validation setup.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a historical stock price DataFrame.

        Args:
            data (pd.DataFrame): Must contain 'Close' and 'Volume' columns.
        """
        self._validate_inputs(data)
        self.data = data.copy().sort_index()
        self.target_horizon = 5 # Default horizon

    def _validate_inputs(self, data: pd.DataFrame) -> None:
        """Validate input dataframe format and contents."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")
        # If columns are multi-indexed (common in yfinance download), flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        required_cols = {"Close", "Volume"}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"data must contain columns: {required_cols}")
        if data.empty:
            raise ValueError("data DataFrame cannot be empty.")
        if len(data) < 60:
            raise ValueError("data DataFrame must contain at least 60 trading days.")

    def engineer_features(self, target_horizon: int = 5) -> tuple[pd.DataFrame, pd.Series]:
        """
        Engineers features for volatility forecasting, including lagged volatilities,
        realized volatility, returns, Parkinson range-based volatility, and rolling moments.

        Args:
            target_horizon (int): The forward horizon (in days) to predict realized volatility for.

        Returns:
            tuple: (X_features, y_target) after dropping NaN values.
        """
        self.target_horizon = int(target_horizon)
        df = self.data.copy()
        
        # Calculate daily log returns
        df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
        
        # 1. Realized Volatilities (annualized)
        df["vol_1"] = np.abs(df["returns"]) * math.sqrt(252.0)
        df["vol_5"] = df["returns"].rolling(window=5).std() * math.sqrt(252.0)
        df["vol_10"] = df["returns"].rolling(window=10).std() * math.sqrt(252.0)
        df["vol_21"] = df["returns"].rolling(window=21).std() * math.sqrt(252.0)
        
        # 2. Parkinson Volatility (range-based estimator)
        has_ohlc = {"High", "Low"}.issubset(df.columns)
        if has_ohlc:
            log_hl = np.log(df["High"] / df["Low"])
            df["vol_parkinson"] = np.sqrt(252.0 * (log_hl ** 2).rolling(window=21).mean() / (4.0 * math.log(2.0)))
        else:
            df["vol_parkinson"] = df["vol_21"].copy()
            
        # 3. Rolling skewness and kurtosis
        df["skew_21"] = df["returns"].rolling(window=21).skew()
        df["kurt_21"] = df["returns"].rolling(window=21).kurt()
        
        # Target: Realized volatility over the next 'target_horizon' days (annualized)
        realized_vol_horizon = df["returns"].rolling(window=self.target_horizon).std() * math.sqrt(252.0)
        df["target"] = realized_vol_horizon.shift(-self.target_horizon)

        # Feature columns list
        feature_cols = [
            "returns", "vol_1", "vol_5", "vol_10", "vol_21",
            "vol_parkinson", "skew_21", "kurt_21"
        ]

        # Drop NaNs
        clean_df = df[feature_cols + ["target"]].dropna()

        X = clean_df[feature_cols]
        y = clean_df["target"]
        
        return X, y

    def train_and_evaluate(
        self,
        X: pd.DataFrame = None,
        y: pd.Series = None,
        train_size: float = 0.70,
        step_size: int = 5
    ) -> dict[str, dict]:
        """
        Perform a rolling walk-forward validation for Historical Baseline, EWMA,
        GARCH(1,1), HAR-RV, Linear Regression, Random Forest, and XGBoost models.

        Args:
            X (pd.DataFrame): Ignored, features recalculated internally to ensure proper alignment.
            y (pd.Series): Ignored, target recalculated internally.
            train_size (float): Proportion of data used for initial training window.
            step_size (int): Re-calibration frequency in days (default 5).

        Returns:
            dict: Performance metrics (RMSE, MAE, R2) and predictions for all models.
        """
        # Re-engineer features internally to ensure returns, High/Low data, and EWMA are perfectly aligned
        df = self.data.copy()
        df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
        
        # Realized volatilities
        df["vol_1"] = np.abs(df["returns"]) * math.sqrt(252.0)
        df["vol_5"] = df["returns"].rolling(window=5).std() * math.sqrt(252.0)
        df["vol_10"] = df["returns"].rolling(window=10).std() * math.sqrt(252.0)
        df["vol_21"] = df["returns"].rolling(window=21).std() * math.sqrt(252.0)
        
        # Parkinson Volatility
        has_ohlc = {"High", "Low"}.issubset(df.columns)
        if has_ohlc:
            log_hl = np.log(df["High"] / df["Low"])
            df["vol_parkinson"] = np.sqrt(252.0 * (log_hl ** 2).rolling(window=21).mean() / (4.0 * math.log(2.0)))
        else:
            df["vol_parkinson"] = df["vol_21"].copy()
            
        # EWMA Volatility (RiskMetrics lambda=0.94)
        decay = 0.94
        variance = np.zeros(len(df))
        variance[0] = np.var(df["returns"].dropna())
        ret_vals = df["returns"].fillna(0.0).values
        for i in range(1, len(df)):
            variance[i] = decay * variance[i-1] + (1.0 - decay) * (ret_vals[i] ** 2)
        df["ewma_vol"] = np.sqrt(variance) * math.sqrt(252.0)
        
        # Moments
        df["skew_21"] = df["returns"].rolling(window=21).skew()
        df["kurt_21"] = df["returns"].rolling(window=21).kurt()
        
        # Target (shifted back)
        realized_vol_horizon = df["returns"].rolling(window=self.target_horizon).std() * math.sqrt(252.0)
        df["target"] = realized_vol_horizon.shift(-self.target_horizon)
        
        # Columns to keep
        feature_cols = [
            "returns", "vol_1", "vol_5", "vol_10", "vol_21",
            "vol_parkinson", "skew_21", "kurt_21"
        ]
        
        clean_df = df[feature_cols + ["target", "ewma_vol"]].dropna()
        if len(clean_df) < 40:
            raise ValueError("Not enough clean data after removing NaNs for rolling validation.")
            
        X_clean = clean_df[feature_cols]
        y_clean = clean_df["target"]
        ewma_clean = clean_df["ewma_vol"]
        returns_clean = clean_df["returns"]
        
        n_samples = len(X_clean)
        start_idx = int(n_samples * train_size)
        
        # Prediction lists
        pred_hist = []
        pred_ewma = []
        pred_garch = []
        pred_har = []
        pred_lr = []
        pred_rf = []
        pred_xgb = []
        
        actuals = []
        test_dates = []
        
        # Walk-forward validation loop
        for t in range(start_idx, n_samples, step_size):
            end_t = min(t + step_size, n_samples)
            
            # Splitting sets
            X_train = X_clean.iloc[:t]
            y_train = y_clean.iloc[:t]
            
            X_test = X_clean.iloc[t : end_t]
            y_test = y_clean.iloc[t : end_t]
            
            if len(X_test) == 0:
                break
                
            actuals.extend(y_test.values)
            test_dates.extend(y_test.index)
            
            # 1. Historical Volatility Baseline (last 21 days realized vol)
            pred_hist.extend(X_test["vol_21"].values)
            
            # 2. EWMA Volatility Model
            pred_ewma.extend(ewma_clean.iloc[t : end_t].values)
            
            # 3. GARCH(1,1) model (using ARCH package)
            # Recalibrate GARCH model once per step block on training set
            train_returns = returns_clean.iloc[:t]
            try:
                # Scale returns by 100 for stable GARCH optimization convergence
                garch_model = arch_model(train_returns * 100.0, p=1, q=1, vol="Garch", dist="normal")
                res = garch_model.fit(update_freq=0, disp="off")
                
                omega = res.params["omega"] / 10000.0 # scale back variance parameters
                alpha = res.params["alpha[1]"]
                beta = res.params["beta[1]"]
                
                # Unconditional variance
                v_uncond = omega / (1.0 - alpha - beta) if (1.0 - alpha - beta) > 0.0 else np.var(train_returns)
                
                # Update loop inside the block to track daily conditional variance
                h_var = (res.conditional_volatility.iloc[-1] / 100.0) ** 2
                block_garch = []
                
                for idx_test in range(len(X_test)):
                    # Compute H-step ahead expected variances
                    # E_t[V_{t+k}] = V_uncond + (alpha+beta)^k * (V_t - V_uncond)
                    persistence = alpha + beta
                    temp_var_sum = 0.0
                    for k in range(1, self.target_horizon + 1):
                        temp_var_sum += v_uncond + (persistence ** k) * (h_var - v_uncond)
                    
                    avg_var = temp_var_sum / self.target_horizon
                    block_garch.append(math.sqrt(max(1e-6, avg_var) * 252.0))
                    
                    # Update variance for next day in block
                    r_actual = returns_clean.iloc[t + idx_test]
                    h_var = omega + alpha * (r_actual ** 2) + beta * h_var
                    
                pred_garch.extend(block_garch)
            except Exception as e:
                # Fallback to EWMA
                logger.warning(f"GARCH calibration failed at step {t}, falling back to EWMA: {e}")
                pred_garch.extend(ewma_clean.iloc[t : end_t].values)
                
            # 4. HAR-RV Model
            har_features = ["vol_1", "vol_5", "vol_21"]
            har_model = LinearRegression()
            har_model.fit(X_train[har_features], y_train)
            pred_har.extend(har_model.predict(X_test[har_features]))
            
            # 5. Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            pred_lr.extend(lr_model.predict(X_test))
            
            # 6. Random Forest
            rf_model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            pred_rf.extend(rf_model.predict(X_test))
            
            # 7. XGBoost
            xgb_model = XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            pred_xgb.extend(xgb_model.predict(X_test))

        actuals = np.array(actuals)
        test_index = pd.Index(test_dates)
        
        predictions_dict = {
            "Historical Baseline": np.array(pred_hist),
            "EWMA Volatility": np.array(pred_ewma),
            "GARCH(1,1)": np.array(pred_garch),
            "HAR-RV Model": np.array(pred_har),
            "Linear Regression": np.array(pred_lr),
            "Random Forest": np.array(pred_rf),
            "XGBoost": np.array(pred_xgb)
        }
        
        results = {}
        for name, preds in predictions_dict.items():
            rmse = math.sqrt(mean_squared_error(actuals, preds))
            mae = mean_absolute_error(actuals, preds)
            r2 = r2_score(actuals, preds)
            
            results[name] = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "predictions": pd.Series(preds, index=test_index),
                "actuals": pd.Series(actuals, index=test_index),
                "model_instance": None
            }
            
        return results
