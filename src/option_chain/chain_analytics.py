import datetime
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from src.pricing.black_scholes import BlackScholes, OptionType
from src.greeks.greeks import Greeks
from src.volatility.implied_vol import ImpliedVolatilitySolver

logger = logging.getLogger(__name__)

class OptionChainAnalytics:
    """
    Downloads and analyzes option chains using yfinance,
    computing Implied Volatilities, Greeks, and ranking contracts.
    """

    def __init__(self, symbol: str, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        self.symbol = symbol.upper()
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.ticker = yf.Ticker(self.symbol)

    def get_current_spot(self) -> float:
        """Fetch the current stock price of the underlying asset."""
        try:
            hist = self.ticker.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
            
            # Fallback to info
            info = self.ticker.info
            spot = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
            if spot is not None:
                return float(spot)
        except Exception as e:
            logger.error(f"Failed to fetch current spot price for {self.symbol}: {e}")
        
        raise ValueError(f"Could not retrieve spot price for symbol {self.symbol}.")

    def get_expirations(self) -> list[str]:
        """Fetch available option expiration dates."""
        try:
            return list(self.ticker.options)
        except Exception as e:
            logger.error(f"Failed to fetch expiration dates for {self.symbol}: {e}")
            return []

    def analyze_chain(self, expiration_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download, compute IVs/Greeks, and analyze the option chain for a given expiration.

        Args:
            expiration_date (str): Expiration date string (YYYY-MM-DD).

        Returns:
            tuple: (calls_df, puts_df) with computed IVs and Greeks.
        """
        # Fetch option chain from yfinance
        try:
            chain = self.ticker.option_chain(expiration_date)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
        except Exception as e:
            logger.error(f"Failed to fetch option chain for {self.symbol} on date {expiration_date}: {e}")
            raise

        spot = self.get_current_spot()

        # Compute time to maturity
        exp_dt = datetime.datetime.strptime(expiration_date, "%Y-%m-%d").date()
        today = datetime.date.today()
        days_to_mat = (exp_dt - today).days
        
        # Set a minimum T (e.g. 1 day) to prevent division by zero in pricing models
        T = max(1, days_to_mat) / 365.0

        # Process Calls and Puts
        calls = self._process_dataframe(calls, spot, T, OptionType.CALL)
        puts = self._process_dataframe(puts, spot, T, OptionType.PUT)

        return calls, puts

    def _process_dataframe(self, df: pd.DataFrame, spot: float, T: float, option_type: OptionType) -> pd.DataFrame:
        """Compute mid prices, Implied Volatilities, and Greeks for a chain DataFrame."""
        if df.empty:
            return df

        # Fill missing bids/asks
        df["bid"] = df["bid"].fillna(0.0)
        df["ask"] = df["ask"].fillna(0.0)
        df["lastPrice"] = df["lastPrice"].fillna(0.0)

        # Mid price calculation
        df["mid_price"] = (df["bid"] + df["ask"]) / 2.0
        # If mid price is zero, fallback to last price
        zero_mid_mask = df["mid_price"] <= 0.0
        df.loc[zero_mid_mask, "mid_price"] = df.loc[zero_mid_mask, "lastPrice"]

        calculated_ivs = []
        bid_ivs = []
        ask_ivs = []
        vol_spreads = []
        deltas = []
        gammas = []
        vegas = []
        thetas = []
        rhos = []

        for idx, row in df.iterrows():
            strike = row["strike"]
            market_price = row["mid_price"]
            
            iv = np.nan
            bid_iv = np.nan
            ask_iv = np.nan
            delta = np.nan
            gamma = np.nan
            vega = np.nan
            theta = np.nan
            rho = np.nan

            if market_price > 0.0:
                try:
                    # Solve for IV
                    iv = ImpliedVolatilitySolver.calculate_iv(
                        market_price=market_price,
                        spot=spot,
                        strike=strike,
                        time_to_maturity=T,
                        risk_free_rate=self.risk_free_rate,
                        option_type=option_type,
                        dividend_yield=self.dividend_yield
                    )
                    
                    # Compute analytical Greeks
                    bs = BlackScholes(
                        spot=spot,
                        strike=strike,
                        time_to_maturity=T,
                        risk_free_rate=self.risk_free_rate,
                        volatility=iv,
                        dividend_yield=self.dividend_yield
                    )
                    greeks_calc = Greeks(bs)
                    
                    delta = greeks_calc.delta(option_type)
                    gamma = greeks_calc.gamma()
                    vega = greeks_calc.vega()
                    theta = greeks_calc.theta(option_type)
                    rho = greeks_calc.rho(option_type)
                except Exception:
                    # If local IV calculation fails, fall back to yfinance's impliedVolatility
                    yf_iv = row.get("impliedVolatility")
                    if yf_iv is not None and yf_iv > 0.0:
                        iv = yf_iv
                        try:
                            bs = BlackScholes(
                                spot=spot,
                                strike=strike,
                                time_to_maturity=T,
                                risk_free_rate=self.risk_free_rate,
                                volatility=iv,
                                dividend_yield=self.dividend_yield
                            )
                            greeks_calc = Greeks(bs)
                            delta = greeks_calc.delta(option_type)
                            gamma = greeks_calc.gamma()
                            vega = greeks_calc.vega()
                            theta = greeks_calc.theta(option_type)
                            rho = greeks_calc.rho(option_type)
                        except Exception:
                            pass

            # Calculate Bid IV
            if row["bid"] > 0.0:
                try:
                    bid_iv = ImpliedVolatilitySolver.calculate_iv(
                        market_price=row["bid"],
                        spot=spot,
                        strike=strike,
                        time_to_maturity=T,
                        risk_free_rate=self.risk_free_rate,
                        option_type=option_type,
                        dividend_yield=self.dividend_yield
                    )
                except Exception:
                    pass

            # Calculate Ask IV
            if row["ask"] > 0.0:
                try:
                    ask_iv = ImpliedVolatilitySolver.calculate_iv(
                        market_price=row["ask"],
                        spot=spot,
                        strike=strike,
                        time_to_maturity=T,
                        risk_free_rate=self.risk_free_rate,
                        option_type=option_type,
                        dividend_yield=self.dividend_yield
                    )
                except Exception:
                    pass

            vol_spread = ask_iv - bid_iv if (not np.isnan(ask_iv) and not np.isnan(bid_iv)) else np.nan

            calculated_ivs.append(iv)
            bid_ivs.append(bid_iv)
            ask_ivs.append(ask_iv)
            vol_spreads.append(vol_spread)
            deltas.append(delta)
            gammas.append(gamma)
            vegas.append(vega)
            thetas.append(theta)
            rhos.append(rho)

        df["computed_iv"] = calculated_ivs
        df["bid_iv"] = bid_ivs
        df["ask_iv"] = ask_ivs
        df["vol_spread"] = vol_spreads
        df["delta"] = deltas
        df["gamma"] = gammas
        df["vega"] = vegas
        df["theta"] = thetas
        df["rho"] = rhos

        # Volume / Open Interest ratio
        df["vol_oi_ratio"] = np.where(df["openInterest"] > 0, df["volume"] / df["openInterest"], np.nan)
        # Bid-Ask spread
        df["bid_ask_spread"] = df["ask"] - df["bid"]
        df["spread_pct"] = np.where(df["mid_price"] > 0, df["bid_ask_spread"] / df["mid_price"] * 100, np.nan)

        return df

    def get_open_interest_summary(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> dict:
        """
        Aggregate Open Interest analytics from calls and puts dataframes.
        """
        total_call_oi = int(calls_df["openInterest"].sum()) if not calls_df.empty else 0
        total_put_oi = int(puts_df["openInterest"].sum()) if not puts_df.empty else 0
        
        pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else np.nan

        # Strike with max OI
        max_call_oi_row = calls_df.loc[calls_df["openInterest"].idxmax()] if not calls_df.empty and calls_df["openInterest"].max() > 0 else None
        max_put_oi_row = puts_df.loc[puts_df["openInterest"].idxmax()] if not puts_df.empty and puts_df["openInterest"].max() > 0 else None

        return {
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "put_call_oi_ratio": pc_ratio,
            "max_call_oi_strike": max_call_oi_row["strike"] if max_call_oi_row is not None else None,
            "max_call_oi_volume": int(max_call_oi_row["openInterest"]) if max_call_oi_row is not None else None,
            "max_put_oi_strike": max_put_oi_row["strike"] if max_put_oi_row is not None else None,
            "max_put_oi_volume": int(max_put_oi_row["openInterest"]) if max_put_oi_row is not None else None,
        }

    def get_volume_summary(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> dict:
        """
        Aggregate Volume analytics from calls and puts dataframes.
        """
        total_call_vol = int(calls_df["volume"].sum()) if not calls_df.empty else 0
        total_put_vol = int(puts_df["volume"].sum()) if not puts_df.empty else 0
        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else np.nan

        # Highest volume contract
        max_call_v_row = calls_df.loc[calls_df["volume"].idxmax()] if not calls_df.empty and calls_df["volume"].max() > 0 else None
        max_put_v_row = puts_df.loc[puts_df["volume"].idxmax()] if not puts_df.empty and puts_df["volume"].max() > 0 else None

        return {
            "total_call_volume": total_call_vol,
            "total_put_volume": total_put_vol,
            "put_call_vol_ratio": pc_ratio,
            "max_call_vol_strike": max_call_v_row["strike"] if max_call_v_row is not None else None,
            "max_call_vol_volume": int(max_call_v_row["volume"]) if max_call_v_row is not None else None,
            "max_put_vol_strike": max_put_v_row["strike"] if max_put_v_row is not None else None,
            "max_put_vol_volume": int(max_put_v_row["volume"]) if max_put_v_row is not None else None,
        }

    def rank_contracts(self, df: pd.DataFrame, by: str = "liquidity", ascending: bool = False) -> pd.DataFrame:
        """
        Rank options contracts based on liquidity (low bid-ask spread and high volume),
        or by analytical Greeks (e.g. highest absolute gamma).

        Args:
            df (pd.DataFrame): Calls or puts DataFrame with analyzed columns.
            by (str): Raking metric: 'liquidity', 'gamma', 'vega', 'volume', or 'open_interest'.
            ascending (bool): Rank ordering.

        Returns:
            pd.DataFrame: Sorted DataFrame copy.
        """
        if df.empty:
            return df

        df_rank = df.copy()

        if by == "liquidity":
            # Rank liquidity: we want low spread % and high volume
            # We can create a composite score or simply sort by spread_pct (ascending=True) and volume (ascending=False)
            # Sorting primarily by lowest spread % (which represents tightest markets)
            return df_rank.sort_values(by=["spread_pct", "volume"], ascending=[True, False])
        elif by == "gamma":
            # Sort by absolute gamma descending
            df_rank["abs_gamma"] = df_rank["gamma"].abs()
            return df_rank.sort_values(by="abs_gamma", ascending=ascending)
        elif by == "vega":
            df_rank["abs_vega"] = df_rank["vega"].abs()
            return df_rank.sort_values(by="abs_vega", ascending=ascending)
        elif by == "volume":
            return df_rank.sort_values(by="volume", ascending=ascending)
        elif by == "open_interest":
            return df_rank.sort_values(by="openInterest", ascending=ascending)
        else:
            raise ValueError(f"Unsupported ranking key: {by}")

    def filter_chain(
        self,
        df: pd.DataFrame,
        option_type: OptionType,
        spot: float,
        T: float,
        max_spread_pct: float = 100.0,
        min_volume: int = 0,
        filter_arbitrage: bool = True
    ) -> pd.DataFrame:
        """
        Filter option contracts by bid-ask spread %, minimum volume, and arbitrage boundaries.

        Args:
            df (pd.DataFrame): Option chain DataFrame (calls or puts).
            option_type (OptionType): OptionType.CALL or OptionType.PUT.
            spot (float): Spot price.
            T (float): Time to maturity in years.
            max_spread_pct (float): Maximum bid-ask spread as a percentage of mid-price (default 100.0%).
            min_volume (int): Minimum volume required (default 0).
            filter_arbitrage (bool): If True, filter out options violating arbitrage boundaries.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        if df.empty:
            return df

        df_filtered = df.copy()

        # 1. Minimum volume filter
        if min_volume > 0:
            df_filtered = df_filtered[df_filtered["volume"] >= min_volume]

        # 2. Maximum bid-ask spread % filter
        if max_spread_pct is not None:
            # Keep rows where spread_pct is NaN (i.e. zero/negative mid price) or <= max_spread_pct
            df_filtered = df_filtered[df_filtered["spread_pct"].isna() | (df_filtered["spread_pct"] <= max_spread_pct)]

        # 3. Arbitrage boundary filter
        if filter_arbitrage:
            r = self.risk_free_rate
            q = self.dividend_yield
            
            valid_mask = []
            for idx, row in df_filtered.iterrows():
                strike = row["strike"]
                price = row["mid_price"]
                
                if option_type == OptionType.CALL:
                    # C >= max(S0*e^{-q*T} - K*e^{-r*T}, 0)
                    lower = max(spot * np.exp(-q * T) - strike * np.exp(-r * T), 0.0)
                    # C <= S0*e^{-q*T}
                    upper = spot * np.exp(-q * T)
                else:
                    # P >= max(K*e^{-r*T} - S0*e^{-q*T}, 0)
                    lower = max(strike * np.exp(-r * T) - spot * np.exp(-q * T), 0.0)
                    # P <= K*e^{-r*T}
                    upper = strike * np.exp(-r * T)
                
                # Check with a small tolerance (e.g. 1e-4) to account for numerical rounding
                is_valid = (price >= lower - 1e-4) and (price <= upper + 1e-4)
                valid_mask.append(is_valid)
                
            df_filtered = df_filtered[valid_mask]

        return df_filtered
