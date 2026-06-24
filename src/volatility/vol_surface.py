import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
from scipy.interpolate import griddata
from src.pricing.black_scholes import OptionType
from src.volatility.implied_vol import ImpliedVolatilitySolver

logger = logging.getLogger(__name__)

class VolatilitySurface:
    """
    Analytics engine to construct and visualize 3D implied volatility surfaces
    (Strike vs. Maturity vs. Implied Volatility).
    """

    def __init__(self, spot: float, risk_free_rate: float, dividend_yield: float = 0.0):
        if spot <= 0.0:
            raise ValueError("Spot price must be strictly positive.")
        
        self.spot = float(spot)
        self.risk_free_rate = float(risk_free_rate)
        self.dividend_yield = float(dividend_yield)

    def build_surface(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate implied volatility for a table of options.

        Args:
            options_data (pd.DataFrame): Dataframe with columns:
                'strike' (float), 'maturity' (float, in years),
                'market_price' (float), 'option_type' (OptionType or str).

        Returns:
            pd.DataFrame: Copy of options_data with a new 'implied_volatility' column.
        """
        required_cols = {"strike", "maturity", "market_price", "option_type"}
        if not required_cols.issubset(options_data.columns):
            raise ValueError(f"options_data must contain columns: {required_cols}")

        df = options_data.copy()
        ivs = []

        for idx, row in df.iterrows():
            opt_type = OptionType(row["option_type"])
            try:
                iv = ImpliedVolatilitySolver.calculate_iv(
                    market_price=row["market_price"],
                    spot=self.spot,
                    strike=row["strike"],
                    time_to_maturity=row["maturity"],
                    risk_free_rate=self.risk_free_rate,
                    option_type=opt_type,
                    dividend_yield=self.dividend_yield
                )
                ivs.append(iv)
            except Exception as e:
                logger.warning(
                    f"Failed to calculate IV for strike={row['strike']}, "
                    f"maturity={row['maturity']}: {e}"
                )
                ivs.append(np.nan)

        df["implied_volatility"] = ivs
        return df

    def plot_surface_3d(self, options_df: pd.DataFrame, num_grid_points: int = 50) -> go.Figure:
        """
        Create a 3D volatility surface plot. Interpolates data using griddata for smoothness.

        Args:
            options_df (pd.DataFrame): Dataframe containing 'strike', 'maturity', and 'implied_volatility'.
            num_grid_points (int): Grid resolution for interpolation (defaults to 50).

        Returns:
            go.Figure: Plotly Figure.
        """
        # Filter valid vols
        valid_data = options_df.dropna(subset=["implied_volatility"])
        if len(valid_data) < 4:
            raise ValueError("Need at least 4 valid implied volatility data points to interpolate a surface.")

        strikes = valid_data["strike"].values
        maturities = valid_data["maturity"].values
        vols = valid_data["implied_volatility"].values * 100.0 # Convert to percentage

        # Define grid boundaries
        grid_strike = np.linspace(strikes.min(), strikes.max(), num_grid_points)
        grid_maturity = np.linspace(maturities.min(), maturities.max(), num_grid_points)
        grid_strike_mesh, grid_maturity_mesh = np.meshgrid(grid_strike, grid_maturity)

        # Interpolate
        grid_vol = griddata(
            (strikes, maturities),
            vols,
            (grid_strike_mesh, grid_maturity_mesh),
            method="linear" # linear is robust for arbitrary points; fallback to nearest for extrapolation
        )
        
        # Fill any NaNs resulting from extrapolation
        nan_mask = np.isnan(grid_vol)
        if np.any(nan_mask):
            grid_vol_nearest = griddata(
                (strikes, maturities),
                vols,
                (grid_strike_mesh, grid_maturity_mesh),
                method="nearest"
            )
            grid_vol[nan_mask] = grid_vol_nearest[nan_mask]

        fig = go.Figure()

        # Add 3D surface trace
        fig.add_trace(go.Surface(
            x=grid_strike_mesh,
            y=grid_maturity_mesh,
            z=grid_vol,
            colorscale="Viridis",
            colorbar=dict(title="IV (%)"),
            hovertemplate="Strike: %{x:.2f}<br>Maturity: %{y:.2f} yrs<br>IV: %{z:.2f}%<extra></extra>"
        ))

        # Add scattered actual market data points for visual completeness
        fig.add_trace(go.Scatter3d(
            x=strikes,
            y=maturities,
            z=vols,
            mode="markers",
            marker=dict(size=4, color="#EF553B", opacity=0.8),
            name="Market Data Points",
            hovertemplate="Strike: %{x:.2f}<br>Maturity: %{y:.2f} yrs<br>IV: %{z:.2f}%<extra></extra>"
        ))

        fig.update_layout(
            title=dict(
                text="Implied Volatility Surface",
                font=dict(size=18, family="Inter")
            ),
            scene=dict(
                xaxis_title="Strike Price",
                yaxis_title="Maturity (Years)",
                zaxis_title="Implied Volatility (%)",
                xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                zaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            template="plotly_dark",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig
