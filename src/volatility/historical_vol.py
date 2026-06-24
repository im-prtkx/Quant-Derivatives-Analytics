import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class HistoricalVolatility:
    """
    Analytics engine for calculating historical rolling volatility and
    detecting volatility regimes from historical stock prices.
    """

    def __init__(self, prices: pd.Series):
        """
        Initialize with a series of historical prices.

        Args:
            prices (pd.Series): Historical asset prices.
        """
        self._validate_inputs(prices)
        # Ensure the series is sorted by index (typically date)
        self.prices = prices.sort_index()

    def _validate_inputs(self, prices: pd.Series) -> None:
        """Validates that inputs are non-empty and numeric."""
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series.")
        if prices.empty:
            raise ValueError("prices series cannot be empty.")
        if prices.isnull().all():
            raise ValueError("prices series cannot contain only NaN values.")
        if (prices < 0.0).any():
            raise ValueError("prices cannot contain negative values.")

    def calculate_returns(self) -> pd.Series:
        """
        Calculate rolling daily log returns.

        r_t = ln(S_t / S_{t-1})

        Returns:
            pd.Series: Daily log returns.
        """
        # Exclude zeros/negatives from log
        valid_prices = self.prices.copy()
        valid_prices[valid_prices <= 0.0] = np.nan
        log_prices = np.log(valid_prices)
        returns = log_prices.diff()
        return returns

    def rolling_volatility(self, window: int = 20, trading_days: int = 252) -> pd.Series:
        """
        Calculate rolling annualized historical volatility.

        Args:
            window (int): Rolling window size in days (defaults to 20).
            trading_days (int): Number of trading days in a year (defaults to 252).

        Returns:
            pd.Series: Rolling annualized volatility.
        """
        if window < 2:
            raise ValueError("Rolling window must be at least 2 days.")
        
        returns = self.calculate_returns()
        # Rolling standard deviation of daily log returns, annualized
        rolling_std = returns.rolling(window=window).std()
        ann_vol = rolling_std * np.sqrt(trading_days)
        return ann_vol

    def detect_regimes(
        self,
        window: int = 20,
        low_threshold: float = 0.25,
        high_threshold: float = 0.75
    ) -> pd.Series:
        """
        Detect volatility regimes ('Low', 'Medium', 'High') based on the rolling
        volatility percentiles over the full series.

        Args:
            window (int): Rolling window size for volatility calculation.
            low_threshold (float): Quantile threshold for 'Low' regime (default: 0.25).
            high_threshold (float): Quantile threshold for 'High' regime (default: 0.75).

        Returns:
            pd.Series: Categorical volatility regimes.
        """
        if not (0.0 < low_threshold < high_threshold < 1.0):
            raise ValueError("Thresholds must satisfy 0 < low_threshold < high_threshold < 1.")

        vol = self.rolling_volatility(window=window)
        
        # Calculate thresholds on non-NaN values
        valid_vols = vol.dropna()
        if valid_vols.empty:
            # Not enough data to calculate volatility
            regimes = pd.Series(index=vol.index, dtype='object')
            return regimes

        q_low = valid_vols.quantile(low_threshold)
        q_high = valid_vols.quantile(high_threshold)

        regimes = pd.Series(index=vol.index, dtype='object')
        
        # Apply classifications where volatility is not NaN
        not_null = vol.notnull()
        regimes[not_null & (vol < q_low)] = "Low"
        regimes[not_null & (vol >= q_low) & (vol <= q_high)] = "Medium"
        regimes[not_null & (vol > q_high)] = "High"

        return regimes
