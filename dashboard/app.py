import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import yfinance as yf

# Pricing & Greeks
from src.pricing.black_scholes import BlackScholes, OptionType
from src.greeks.greeks import Greeks
from src.pricing.binomial_tree import BinomialTree
from src.pricing.monte_carlo import MonteCarloEngine

# Volatility
from src.volatility.implied_vol import ImpliedVolatilitySolver
from src.volatility.historical_vol import HistoricalVolatility
from src.volatility.vol_smile import VolatilitySmile, sabr_volatility
from src.volatility.vol_surface import VolatilitySurface
from src.volatility.vol_forecasting import VolatilityForecaster

# Strategies, Portfolio, Option Chain
from src.strategies.strategy import (
    LongCall, LongPut, CoveredCall, ProtectivePut,
    BullCallSpread, BearPutSpread, IronCondor,
    LongStraddle, LongStrangle
)
from src.portfolio.portfolio import PortfolioRiskEngine, StockPosition, OptionPosition
from src.option_chain.chain_analytics import OptionChainAnalytics

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Quantitative Derivatives Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        font-family: 'Outfit', 'Inter', sans-serif;
        background: linear-gradient(135deg, #00FFCC 0%, #0099FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 16px;
        color: #A0AEC0;
        font-family: 'Inter', sans-serif;
        margin-bottom: 25px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-title">Quantitative Derivatives Analytics Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A production-grade options pricing, risk management, and volatility modeling engine</div>', unsafe_allow_html=True)

# Define Tabs
tabs = st.tabs([
    "Black-Scholes Pricing",
    "Greeks Sensitivity",
    "Binomial Tree",
    "Monte Carlo Simulation",
    "Implied Volatility Solver",
    "Volatility Smile",
    "Volatility Surface 3D",
    "Strategy Payoff Builder",
    "Portfolio Risk & Stress",
    "Option Chain Analytics",
    "Volatility Forecasting ML"
])

# Sidebar global parameters (or defaults)
st.sidebar.header("Global Constants")
spot_default = st.sidebar.number_input("Underlying Spot Price ($)", value=100.0, min_value=0.01, step=1.0)
strike_default = st.sidebar.number_input("Option Strike Price ($)", value=100.0, min_value=0.01, step=1.0)
maturity_default = st.sidebar.number_input("Time to Maturity (Years)", value=0.5, min_value=0.001, max_value=10.0, step=0.1)
r_default = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, min_value=-0.1, max_value=0.5, step=0.01)
vol_default = st.sidebar.number_input("Volatility (σ)", value=0.20, min_value=0.01, max_value=2.0, step=0.01)
div_default = st.sidebar.number_input("Dividend Yield (q)", value=0.0, min_value=0.0, max_value=0.3, step=0.01)

# Initialize Session State for Portfolio
if "portfolio_positions" not in st.session_state:
    # Set default positions (Stock + Call Spread)
    st.session_state.portfolio_positions = [
        {"type": "Stock", "qty": 100.0, "spot": 100.0},
        {"type": "Option", "opt_type": "call", "strike": 100.0, "maturity": 0.25, "vol": 0.20, "qty": 2.0},
        {"type": "Option", "opt_type": "call", "strike": 105.0, "maturity": 0.25, "vol": 0.22, "qty": -2.0}
    ]


# -------------------------------------------------------------
# TAB 1: Black-Scholes Pricing
# -------------------------------------------------------------
with tabs[0]:
    st.header("Black-Scholes-Merton Pricing Model")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Inputs")
        option_choice = st.selectbox("Option Type", ["CALL", "PUT"], key="bs_opt_type")
        opt_type = OptionType.CALL if option_choice == "CALL" else OptionType.PUT
        
        # Calculate
        bs = BlackScholes(
            spot=spot_default,
            strike=strike_default,
            time_to_maturity=maturity_default,
            risk_free_rate=r_default,
            volatility=vol_default,
            dividend_yield=div_default
        )
        
        price = bs.price(opt_type)
        
        st.metric(label=f"BS European {option_choice} Price", value=f"${price:.4f}")
        
        # Display Greeks
        st.subheader("Analytical Greeks")
        greeks = Greeks(bs)
        delta_val = greeks.delta(opt_type)
        gamma_val = greeks.gamma()
        vega_val = greeks.vega()
        theta_val = greeks.theta(opt_type, annualized=False)
        rho_val = greeks.rho(opt_type)
        
        st.write(f"**Delta (Δ):** `{delta_val:.6f}`")
        st.write(f"**Gamma (γ):** `{gamma_val:.6f}`")
        st.write(f"**Vega (ν):** `{vega_val:.6f}`")
        st.write(f"**Theta (θ, Daily):** `{theta_val:.6f}`")
        st.write(f"**Rho (ρ):** `{rho_val:.6f}`")

    with col2:
        st.subheader("Payoff and P&L at Expiration")
        # Generate payoff grid
        spots = np.linspace(spot_default * 0.5, spot_default * 1.5, 100)
        payoffs = np.maximum(spots - strike_default, 0.0) if opt_type == OptionType.CALL else np.maximum(strike_default - spots, 0.0)
        profits = payoffs - price
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spots, y=profits, name="P&L", line=dict(color="#00CC96", width=3), fill="tozeroy", fillcolor="rgba(0, 204, 150, 0.1)"))
        fig.add_trace(go.Scatter(x=spots, y=payoffs, name="Payoff", line=dict(color="#636EFA", width=1.5, dash="dash")))
        fig.add_hline(y=0, line_color="white", line_dash="solid", line_width=1)
        fig.add_vline(x=spot_default, line_color="#00FFCC", line_dash="dot", annotation_text=f"Spot ({spot_default:.1f})")
        fig.add_vline(x=strike_default + (price if opt_type == OptionType.CALL else -price), line_color="#EF553B", line_dash="dot", annotation_text="Breakeven")
        
        fig.update_layout(xaxis_title="Underlying Spot S_T", yaxis_title="Profit / Payoff ($)", template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------
# TAB 2: Greeks Sensitivity
# -------------------------------------------------------------
with tabs[1]:
    st.header("Option Greeks Sensitivity Analysis")
    st.write("Understand how option sensitivities (Greeks) change as the underlying spot price shifts.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Chart Parameters")
        greek_to_plot = st.selectbox("Greek to Plot", [
            "Delta", "Gamma", "Vega", "Theta (Daily)", "Rho",
            "Vanna", "Volga", "Charm", "Speed", "Color"
        ])
        opt_choice_greek = st.selectbox("Option Type ", ["CALL", "PUT"])
        opt_type_greek = OptionType.CALL if opt_choice_greek == "CALL" else OptionType.PUT
        
        spot_range = np.linspace(spot_default * 0.4, spot_default * 1.6, 100)
        
    with col2:
        # Calculate Greeks across range
        greek_values = []
        for s in spot_range:
            bs_temp = BlackScholes(spot=s, strike=strike_default, time_to_maturity=maturity_default, risk_free_rate=r_default, volatility=vol_default, dividend_yield=div_default)
            g_temp = Greeks(bs_temp)
            
            if greek_to_plot == "Delta":
                greek_values.append(g_temp.delta(opt_type_greek))
            elif greek_to_plot == "Gamma":
                greek_values.append(g_temp.gamma())
            elif greek_to_plot == "Vega":
                greek_values.append(g_temp.vega())
            elif greek_to_plot == "Theta (Daily)":
                greek_values.append(g_temp.theta(opt_type_greek, annualized=False))
            elif greek_to_plot == "Rho":
                greek_values.append(g_temp.rho(opt_type_greek))
            elif greek_to_plot == "Vanna":
                greek_values.append(g_temp.vanna())
            elif greek_to_plot == "Volga":
                greek_values.append(g_temp.volga())
            elif greek_to_plot == "Charm":
                greek_values.append(g_temp.charm(opt_type_greek))
            elif greek_to_plot == "Speed":
                greek_values.append(g_temp.speed())
            elif greek_to_plot == "Color":
                greek_values.append(g_temp.color())
                
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(x=spot_range, y=greek_values, name=greek_to_plot, line=dict(color="#0099FF", width=3)))
        fig_g.add_vline(x=spot_default, line_color="#00FFCC", line_dash="dot", annotation_text=f"Current Spot ({spot_default})")
        fig_g.add_vline(x=strike_default, line_color="#EF553B", line_dash="dot", annotation_text="Strike Price")
        fig_g.update_layout(xaxis_title="Underlying Spot Price", yaxis_title=greek_to_plot, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_g, use_container_width=True)


# -------------------------------------------------------------
# TAB 3: Binomial Tree
# -------------------------------------------------------------
with tabs[2]:
    st.header("Binomial Tree Option Pricing (CRR vs. Leisen-Reimer)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Configuration")
        tree_steps = st.slider("Number of Steps (N)", min_value=1, max_value=200, value=50, step=1)
        tree_opt_type = st.selectbox("Option Type  ", ["CALL", "PUT"])
        tree_style = st.selectbox("Execution Style", ["American", "European"])
        model_type_choice = st.selectbox("Model Type", ["LR (Leisen-Reimer)", "CRR (Cox-Ross-Rubinstein)"])
        
        is_american = (tree_style == "American")
        opt_type_tree = OptionType.CALL if tree_opt_type == "CALL" else OptionType.PUT
        model_type = "lr" if model_type_choice.startswith("LR") else "crr"
        
        tree = BinomialTree(
            spot=spot_default,
            strike=strike_default,
            time_to_maturity=maturity_default,
            risk_free_rate=r_default,
            volatility=vol_default,
            dividend_yield=div_default,
            steps=tree_steps,
            model_type=model_type
        )
        
        tree_price = tree.price(opt_type_tree, american=is_american)
        st.metric(label=f"Binomial Tree Option Price ({model_type.upper()})", value=f"${tree_price:.4f}")
        
        # American Early Exercise Boundary Plot
        if is_american:
            st.write("**Early Exercise Boundary (S*)**")
            boundary = tree.get_early_exercise_boundary(opt_type_tree)
            times = [item[0] for item in boundary]
            spots_boundary = [item[1] for item in boundary]
            
            times_clean = [t for t, s in zip(times, spots_boundary) if s is not None]
            spots_clean = [s for t, s in zip(times, spots_boundary) if s is not None]
            
            if spots_clean:
                fig_boundary = go.Figure()
                fig_boundary.add_trace(go.Scatter(x=times_clean, y=spots_clean, mode="lines+markers", name="Exercise Boundary", line=dict(color="#FF3366", width=2.5)))
                fig_boundary.add_hline(y=strike_default, line_color="white", line_dash="dash", annotation_text="Strike Price")
                fig_boundary.update_layout(
                    xaxis_title="Time (Years)",
                    yaxis_title="Stock Price S*(t)",
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_boundary, use_container_width=True)
            else:
                st.info("No early exercise nodes found. Option behaves like European.")
        
    with col2:
        st.subheader("Binomial Pricing Convergence")
        if st.button("Run Convergence Comparison"):
            with st.spinner("Calculating tree convergence..."):
                # Compute CRR convergence
                crr_tree_conv = BinomialTree(
                    spot=spot_default, strike=strike_default, time_to_maturity=maturity_default,
                    risk_free_rate=r_default, volatility=vol_default, dividend_yield=div_default,
                    steps=tree_steps, model_type="crr"
                )
                conv_crr = crr_tree_conv.compare_convergence(
                    option_type=opt_type_tree, american=is_american, max_steps=120, step_increment=4
                )
                
                # Compute LR convergence
                lr_tree_conv = BinomialTree(
                    spot=spot_default, strike=strike_default, time_to_maturity=maturity_default,
                    risk_free_rate=r_default, volatility=vol_default, dividend_yield=div_default,
                    steps=tree_steps, model_type="lr"
                )
                conv_lr = lr_tree_conv.compare_convergence(
                    option_type=opt_type_tree, american=is_american, max_steps=120, step_increment=4
                )
                
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(x=conv_crr["steps"], y=conv_crr["binomial_prices"], name="CRR (Sawtooth)", mode="markers+lines", line=dict(color="#FF9900", width=1.5)))
                fig_c.add_trace(go.Scatter(x=conv_lr["steps"], y=conv_lr["binomial_prices"], name="Leisen-Reimer (Smooth)", mode="markers+lines", line=dict(color="#00CC96", width=2)))
                if not is_american:
                    fig_c.add_trace(go.Scatter(x=conv_crr["steps"], y=conv_crr["bs_prices"], name="Black-Scholes Price", line=dict(color="white", dash="dash", width=1.5)))
                
                fig_c.update_layout(
                    title="Convergence Comparison (CRR vs. Leisen-Reimer)",
                    xaxis_title="Number of Steps (N)",
                    yaxis_title="Option Price ($)",
                    template="plotly_dark",
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_c, use_container_width=True)


# -------------------------------------------------------------
# TAB 4: Monte Carlo Simulation
# -------------------------------------------------------------
with tabs[3]:
    st.header("Monte Carlo Option Pricing & Path Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Options")
        model_choice = st.radio("Simulation Model", ["Geometric Brownian Motion (GBM)", "Heston Stochastic Volatility"])
        
        mc_paths = st.number_input("Number of Simulated Paths", value=20000, min_value=100, max_value=500000, step=5000)
        mc_steps = st.slider("Time Steps per Path", min_value=5, max_value=200, value=50, step=5)
        mc_opt_choice = st.selectbox("Option Type   ", ["CALL", "PUT"])
        mc_opt_type = OptionType.CALL if mc_opt_choice == "CALL" else OptionType.PUT
        
        engine_mc = MonteCarloEngine(
            spot=spot_default,
            strike=strike_default,
            time_to_maturity=maturity_default,
            risk_free_rate=r_default,
            volatility=vol_default,
            dividend_yield=div_default
        )
        
        if model_choice == "Geometric Brownian Motion (GBM)":
            scheme_choice = st.selectbox("Discretization Scheme", ["Exact Analytical", "Euler-Maruyama", "Milstein"])
            scheme = "exact" if scheme_choice.startswith("Exact") else ("euler" if scheme_choice.endswith("Maruyama") else "milstein")
            
            vr_choice = st.selectbox("Variance Reduction", ["None", "Antithetic Variates", "Control Variates (Spot Price)"])
            vr_method = "none" if vr_choice.startswith("None") else ("antithetic" if vr_choice.startswith("Antithetic") else "control_variates")
            
            if st.button("Execute GBM Monte Carlo"):
                with st.spinner("Simulating GBM paths..."):
                    mc_price, std_err, (ci_l, ci_u) = engine_mc.price(
                        option_type=mc_opt_type,
                        num_paths=mc_paths,
                        num_steps=mc_steps,
                        antithetic=(vr_method == "antithetic"),
                        variance_reduction=vr_method,
                        scheme=scheme,
                        seed=42
                    )
                    
                    st.metric(label="Monte Carlo Price Estimate", value=f"${mc_price:.4f}")
                    st.write(f"**Standard Error (SE):** `{std_err:.6f}`")
                    st.write(f"**95% Confidence Interval:** `[${ci_l:.4f}, ${ci_u:.4f}]`")
                    
                    # Fetch reference analytical price
                    bs_ref = BlackScholes(spot_default, strike_default, maturity_default, r_default, vol_default, div_default)
                    ref_price = bs_ref.price(mc_opt_type)
                    st.write(f"**Analytical BS Reference Price:** `${ref_price:.4f}`")
                    
                    # Cache paths for plotting
                    st.session_state.mc_paths_cache = engine_mc.simulate_paths(
                        num_paths=min(100, mc_paths),
                        num_steps=mc_steps,
                        antithetic=(vr_method == "antithetic"),
                        scheme=scheme,
                        seed=42
                    )
                    st.session_state.mc_run_success = True
                    st.session_state.mc_model_type = "gbm"
        else:
            # Heston Parameters
            st.write("**Heston Parameters**")
            h_v0 = st.number_input("Initial Variance (v0)", value=0.04, min_value=0.0001, max_value=2.0, step=0.01)
            h_kappa = st.number_input("Mean Reversion Speed (kappa)", value=2.0, min_value=0.01, max_value=20.0, step=0.5)
            h_theta = st.number_input("Long-Term Variance (theta)", value=0.04, min_value=0.0001, max_value=2.0, step=0.01)
            h_xi = st.number_input("Volatility of Vol (xi)", value=0.3, min_value=0.01, max_value=2.0, step=0.05)
            h_rho = st.slider("correlation (rho)", min_value=-0.99, max_value=0.99, value=-0.70, step=0.05)
            
            if st.button("Execute Heston Monte Carlo"):
                with st.spinner("Simulating Heston joint paths..."):
                    mc_price, std_err = engine_mc.price_heston(
                        option_type=mc_opt_type,
                        num_paths=mc_paths,
                        num_steps=mc_steps,
                        v0=h_v0,
                        kappa=h_kappa,
                        theta=h_theta,
                        xi=h_xi,
                        rho=h_rho,
                        seed=42
                    )
                    
                    st.metric(label="Heston Monte Carlo Price", value=f"${mc_price:.4f}")
                    st.write(f"**Standard Error (SE):** `{std_err:.6f}`")
                    st.write(f"**95% Confidence Interval:** `[${mc_price - 1.96 * std_err:.4f}, ${mc_price + 1.96 * std_err:.4f}]`")
                    
                    # Cache paths
                    S_paths, v_paths = engine_mc.simulate_heston(
                        num_paths=min(100, mc_paths),
                        num_steps=mc_steps,
                        v0=h_v0,
                        kappa=h_kappa,
                        theta=h_theta,
                        xi=h_xi,
                        rho=h_rho,
                        seed=42
                    )
                    st.session_state.mc_paths_cache = S_paths
                    st.session_state.mc_var_paths_cache = v_paths
                    st.session_state.mc_run_success = True
                    st.session_state.mc_model_type = "heston"
        
    with col2:
        st.subheader("Simulated Price Trajectories")
        if st.session_state.get("mc_run_success"):
            paths = st.session_state.mc_paths_cache
            t_grid = np.linspace(0, maturity_default, mc_steps + 1)
            
            if st.session_state.mc_model_type == "gbm":
                fig_mc = go.Figure()
                for path_idx in range(min(50, paths.shape[0])):
                    fig_mc.add_trace(go.Scatter(x=t_grid, y=paths[path_idx], mode="lines", opacity=0.4, showlegend=False, line=dict(width=1)))
                fig_mc.update_layout(xaxis_title="Time (Years)", yaxis_title="Stock Price ($)", template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_mc, use_container_width=True)
            else:
                # Plot Heston joint Spot & Volatility paths
                fig_spot = go.Figure()
                fig_vol = go.Figure()
                v_paths = st.session_state.mc_var_paths_cache
                
                for path_idx in range(min(50, paths.shape[0])):
                    fig_spot.add_trace(go.Scatter(x=t_grid, y=paths[path_idx], mode="lines", opacity=0.4, showlegend=False, line=dict(width=1)))
                    fig_vol.add_trace(go.Scatter(x=t_grid, y=np.sqrt(v_paths[path_idx]) * 100, mode="lines", opacity=0.4, showlegend=False, line=dict(width=1)))
                
                fig_spot.update_layout(title="Asset Price Paths S(t)", xaxis_title="Time", yaxis_title="Stock Price ($)", template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
                fig_vol.update_layout(title="Stochastic Volatility Paths σ(t) %", xaxis_title="Time", yaxis_title="Volatility (%)", template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
                
                st.plotly_chart(fig_spot, use_container_width=True)
                st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Execute Monte Carlo pricing simulation to generate and view price paths.")


# -------------------------------------------------------------
# TAB 5: Implied Volatility Solver
# -------------------------------------------------------------
with tabs[4]:
    st.header("Implied Volatility (IV) Numerical Solver")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Solver Inputs")
        mkt_price = st.number_input("Observed Option Market Price ($)", value=8.0, min_value=0.01, step=0.5)
        iv_opt_choice = st.selectbox("Option Type    ", ["CALL", "PUT"])
        iv_opt_type = OptionType.CALL if iv_opt_choice == "CALL" else OptionType.PUT
        solver_method = st.selectbox("Solver Method", ["Newton-Raphson", "Halley's Method"])
        method_key = "newton" if solver_method.startswith("Newton") else "halley"
        
        if st.button("Solve Implied Volatility"):
            try:
                solved_iv = ImpliedVolatilitySolver.calculate_iv(
                    market_price=mkt_price,
                    spot=spot_default,
                    strike=strike_default,
                    time_to_maturity=maturity_default,
                    risk_free_rate=r_default,
                    option_type=iv_opt_type,
                    dividend_yield=div_default,
                    method=method_key
                )
                st.success("Solver converged successfully!")
                st.metric(label="Calculated Implied Volatility (IV)", value=f"{solved_iv * 100:.3f}%")
            except ValueError as ve:
                st.error(f"Solver Error: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    with col2:
        st.subheader("Mathematical Arbitrage Boundaries")
        st.write("For implied volatility to be mathematically defined, option market prices must sit strictly within arbitrage limits:")
        
        discount_r = math.exp(-r_default * maturity_default)
        discount_q = math.exp(-div_default * maturity_default)
        if iv_opt_type == OptionType.CALL:
            lb = max(spot_default * discount_q - strike_default * discount_r, 0.0)
            ub = spot_default * discount_q
        else:
            lb = max(strike_default * discount_r - spot_default * discount_q, 0.0)
            ub = strike_default * discount_r
            
        st.write(f"- **Lower Boundary (Intrinsic Value):** `${lb:.4f}`")
        st.write(f"- **Upper Boundary (Maximum Value):** `${ub:.4f}`")
        st.write(f"- **Current Option Price Entered:** `${mkt_price:.4f}`")
        
        if lb <= mkt_price <= ub:
            st.success("Entered price lies within valid mathematical boundaries.")
        else:
            st.error("Entered price is outside arbitrage bounds. Solver will not run.")


# -------------------------------------------------------------
# TAB 6: Volatility Smile
# -------------------------------------------------------------
with tabs[5]:
    st.header("Volatility Smile Modeling & SABR Calibration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Smile Dataset Configuration")
        st.write("Modify the default option prices across strikes to observe volatility skew / smiles.")
        
        # Default smile data generator
        smile_strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
        # Simulate realistic skewed call prices for spot 100, r=0.05, T=0.5
        vols_skew = [0.28, 0.25, 0.23, 0.21, 0.20, 0.195, 0.19, 0.192, 0.198]
        
        prices_smile = []
        for k, v in zip(smile_strikes, vols_skew):
            bs_temp = BlackScholes(spot=spot_default, strike=k, time_to_maturity=maturity_default, risk_free_rate=r_default, volatility=v, dividend_yield=div_default)
            prices_smile.append(bs_temp.call_price())
            
        # Display inputs in an editable table format
        smile_df = pd.DataFrame({
            "Strike": smile_strikes,
            "Market Price": prices_smile
        })
        
        edited_df = st.data_editor(smile_df, num_rows="fixed")
        
        # SABR Settings
        st.subheader("SABR Model Fit")
        calibrate_sabr_checkbox = st.checkbox("Calibrate SABR Model", value=True)
        sabr_beta = st.slider("Fixed Beta (Exponent)", 0.0, 1.0, 0.5, 0.1)
        
    with col2:
        st.subheader("Implied Volatility Smile Plot")
        smile_engine = VolatilitySmile(
            spot=spot_default,
            time_to_maturity=maturity_default,
            risk_free_rate=r_default,
            dividend_yield=div_default
        )
        
        processed_smile = smile_engine.calculate_smile(
            strikes=edited_df["Strike"].values,
            market_prices=edited_df["Market Price"].values,
            option_type=OptionType.CALL
        )
        
        sabr_params = None
        if calibrate_sabr_checkbox:
            try:
                alpha_cal, rho_cal, nu_cal = smile_engine.calibrate_sabr(
                    strikes=processed_smile["strike"].values,
                    market_vols=processed_smile["implied_volatility"].values,
                    beta=sabr_beta
                )
                sabr_params = (alpha_cal, rho_cal, nu_cal)
                
                # Display calibrated params
                c1, c2, c3 = st.columns(3)
                c1.metric("Alpha (α)", f"{alpha_cal:.4f}")
                c2.metric("Rho (ρ)", f"{rho_cal:.4f}")
                c3.metric("Nu (ν)", f"{nu_cal:.4f}")
            except Exception as e:
                st.error(f"SABR Calibration Failed: {e}")
        
        fig_s = smile_engine.plot_smile(processed_smile, OptionType.CALL, sabr_params=sabr_params, beta=sabr_beta)
        st.plotly_chart(fig_s, use_container_width=True)


# -------------------------------------------------------------
# TAB 7: Volatility Surface 3D
# -------------------------------------------------------------
with tabs[6]:
    st.header("Implied Volatility Surface 3D Model")
    st.write("A 3D visualization showing Implied Volatility across varying Strike Prices and Expiration Horizons.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Model Specifications")
        vol_skew_magnitude = st.slider("Volatility Skew Steepness", 0.0, 0.5, 0.2, 0.05)
        vol_term_slope = st.slider("Volatility Term Slope (Long Term Vol)", 0.1, 0.4, 0.25, 0.05)
        
    with col2:
        # Generate synthetic surface options
        strikes = [85, 90, 95, 100, 105, 110, 115]
        maturities = [0.1, 0.25, 0.5, 0.75, 1.0]
        
        rows = []
        for k in strikes:
            for t in maturities:
                # Vol = Base (vol_default) + skew term + term structure slope
                skew_term = vol_skew_magnitude * ((100.0 - k) / 30.0)
                term_term = (vol_term_slope - vol_default) * (t / 1.0)
                local_vol = max(0.05, vol_default + skew_term + term_term)
                
                bs_temp = BlackScholes(spot=spot_default, strike=k, time_to_maturity=t, risk_free_rate=r_default, volatility=local_vol, dividend_yield=div_default)
                rows.append({
                    "strike": k,
                    "maturity": t,
                    "market_price": bs_temp.call_price(),
                    "option_type": OptionType.CALL
                })
        
        surf_df = pd.DataFrame(rows)
        surface_engine = VolatilitySurface(spot=spot_default, risk_free_rate=r_default, dividend_yield=div_default)
        
        with st.spinner("Constructing 3D surface model..."):
            solved_surface = surface_engine.build_surface(surf_df)
            fig_surf = surface_engine.plot_surface_3d(solved_surface)
            st.plotly_chart(fig_surf, use_container_width=True)


# -------------------------------------------------------------
# TAB 8: Strategy Payoff Builder
# -------------------------------------------------------------
with tabs[7]:
    st.header("Option Strategy Payoff Builder")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Predefined Strategies")
        strategy_type = st.selectbox("Select Strategy", [
            "Long Call", "Long Put", "Covered Call", "Protective Put",
            "Bull Call Spread", "Bear Put Spread", "Iron Condor",
            "Long Straddle", "Long Strangle"
        ])
        
        # Build fields dynamically depending on selected strategy
        if strategy_type == "Long Call":
            k = st.number_input("Long Call Strike ($)", value=100.0)
            p = st.number_input("Premium paid ($)", value=4.0)
            strat = LongCall(k, p)
        elif strategy_type == "Long Put":
            k = st.number_input("Long Put Strike ($)", value=100.0)
            p = st.number_input("Premium paid ($)", value=3.0)
            strat = LongPut(k, p)
        elif strategy_type == "Covered Call":
            s_entry = st.number_input("Stock Purchase Price ($)", value=100.0)
            k = st.number_input("Short Call Strike ($)", value=105.0)
            p = st.number_input("Call Premium collected ($)", value=2.5)
            strat = CoveredCall(s_entry, k, p)
        elif strategy_type == "Protective Put":
            s_entry = st.number_input("Stock Purchase Price ($)", value=100.0)
            k = st.number_input("Long Put Strike ($)", value=95.0)
            p = st.number_input("Put Premium paid ($)", value=3.0)
            strat = ProtectivePut(s_entry, k, p)
        elif strategy_type == "Bull Call Spread":
            k_long = st.number_input("Long Strike K1 ($)", value=95.0)
            p_long = st.number_input("Long Premium ($)", value=6.0)
            k_short = st.number_input("Short Strike K2 ($)", value=105.0)
            p_short = st.number_input("Short Premium ($)", value=1.5)
            strat = BullCallSpread(k_long, p_long, k_short, p_short)
        elif strategy_type == "Bear Put Spread":
            k_long = st.number_input("Long Strike K2 ($)", value=105.0)
            p_long = st.number_input("Long Premium ($)", value=5.5)
            k_short = st.number_input("Short Strike K1 ($)", value=95.0)
            p_short = st.number_input("Short Premium ($)", value=1.5)
            strat = BearPutSpread(k_long, p_long, k_short, p_short)
        elif strategy_type == "Iron Condor":
            k1 = st.number_input("Long Put Strike K1 ($)", value=90.0)
            p1 = st.number_input("Long Put Premium ($)", value=1.0)
            k2 = st.number_input("Short Put Strike K2 ($)", value=95.0)
            p2 = st.number_input("Short Put Premium ($)", value=2.5)
            k3 = st.number_input("Short Call Strike K3 ($)", value=105.0)
            p3 = st.number_input("Short Call Premium ($)", value=2.5)
            k4 = st.number_input("Long Call Strike K4 ($)", value=110.0)
            p4 = st.number_input("Long Call Premium ($)", value=1.0)
            strat = IronCondor(k1, p1, k2, p2, k3, p3, k4, p4)
        elif strategy_type == "Long Straddle":
            k = st.number_input("Strike ($)", value=100.0)
            c = st.number_input("Call Premium ($)", value=4.0)
            p = st.number_input("Put Premium ($)", value=3.0)
            strat = LongStraddle(k, c, p)
        elif strategy_type == "Long Strangle":
            k_put = st.number_input("Long Put Strike K1 ($)", value=95.0)
            p_put = st.number_input("Put Premium ($)", value=2.5)
            k_call = st.number_input("Long Call Strike K2 ($)", value=105.0)
            p_call = st.number_input("Call Premium ($)", value=3.0)
            strat = LongStrangle(k_put, p_put, k_call, p_call)
            
        metrics = strat.get_metrics()
        
        st.subheader("Strategy Metrics Summary")
        max_p = metrics["max_profit"]
        max_l = metrics["max_loss"]
        max_p_str = f"${max_p:.2f}" if isinstance(max_p, (int, float)) else str(max_p)
        max_l_str = f"${max_l:.2f}" if isinstance(max_l, (int, float)) else str(max_l)
        
        st.metric("Maximum Profit", max_p_str)
        st.metric("Maximum Loss", max_l_str)
        st.write(f"**Breakeven Points:** `{[f'${be:.2f}' for be in metrics.get('breakevens', [])]}`")

    with col2:
        st.subheader("Payoff Profile Chart")
        spot_grid = np.linspace(spot_default * 0.6, spot_default * 1.4, 200)
        fig_strat = strat.plot_payoff(spot_grid)
        st.plotly_chart(fig_strat, use_container_width=True)


# -------------------------------------------------------------
# TAB 9: Portfolio Risk & Stress
# -------------------------------------------------------------
with tabs[8]:
    st.header("Portfolio Risk Engine & Stress Testing")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Portfolio Positions")
        
        # Display current positions
        pos_df = pd.DataFrame(st.session_state.portfolio_positions)
        st.dataframe(pos_df, use_container_width=True)
        
        # Position adding form
        with st.form("add_position_form"):
            st.write("**Add Position**")
            pos_type = st.selectbox("Asset Type", ["Stock", "Option"])
            
            p_qty = st.number_input("Quantity (Positive for Long, Negative for Short)", value=100.0, step=10.0)
            
            p_opt_type = st.selectbox("Option Type (for Options)", ["call", "put"])
            p_strike = st.number_input("Strike Price (for Options)", value=100.0)
            p_mat = st.number_input("Maturity (Years, for Options)", value=0.5)
            p_vol = st.number_input("Volatility (σ, for Options)", value=0.20)
            
            submitted = st.form_submit_button("Add to Portfolio")
            if submitted:
                if pos_type == "Stock":
                    st.session_state.portfolio_positions.append({
                        "type": "Stock",
                        "qty": p_qty,
                        "spot": spot_default
                    })
                else:
                    st.session_state.portfolio_positions.append({
                        "type": "Option",
                        "opt_type": p_opt_type,
                        "strike": p_strike,
                        "maturity": p_mat,
                        "vol": p_vol,
                        "qty": p_qty
                    })
                st.rerun()
                
        if st.button("Clear Portfolio"):
            st.session_state.portfolio_positions = []
            st.rerun()

    with col2:
        st.subheader("Aggregated Portfolio Risk Metrics")
        
        # Construct positions list for engine
        engine_positions = []
        for pos in st.session_state.portfolio_positions:
            if pos["type"] == "Stock":
                engine_positions.append(StockPosition(quantity=pos["qty"]))
            else:
                engine_positions.append(OptionPosition(
                    option_type=OptionType(pos["opt_type"]),
                    strike=pos["strike"],
                    time_to_maturity=pos["maturity"],
                    risk_free_rate=r_default,
                    volatility=pos["vol"],
                    quantity=pos["qty"],
                    dividend_yield=div_default
                ))
                
        if not engine_positions:
            st.warning("Your portfolio is currently empty.")
        else:
            port_engine = PortfolioRiskEngine(spot=spot_default, positions=engine_positions)
            agg_greeks = port_engine.aggregate_greeks()
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Portfolio Delta (Δ)", f"{agg_greeks['delta']:.2f}")
            m2.metric("Portfolio Gamma (γ)", f"{agg_greeks['gamma']:.4f}")
            m3.metric("Portfolio Vega (ν)", f"{agg_greeks['vega']:.2f}")
            m4.metric("Portfolio Theta (θ, Daily)", f"{agg_greeks['theta'] / 365:.2f}")
            m5.metric("Portfolio Rho (ρ)", f"{agg_greeks['rho']:.2f}")
            
            # Value at Risk / Expected Shortfall Section
            st.subheader("Value at Risk (VaR) & Expected Shortfall (ES)")
            col_risk1, col_risk2 = st.columns(2)
            with col_risk1:
                risk_vol = st.number_input("Underlying Daily/Annual Volatility", value=vol_default, min_value=0.01)
                risk_conf = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
            with col_risk2:
                risk_hp = st.number_input("Holding Period (Days)", value=1, min_value=1)
                
            dn_var, dn_es = port_engine.calculate_delta_normal_var_es(risk_vol, risk_conf, risk_hp)
            dg_var, dg_es = port_engine.calculate_delta_gamma_var_es(risk_vol, risk_conf, risk_hp)
            mc_var, mc_es = port_engine.calculate_monte_carlo_var_es(risk_vol, risk_conf, risk_hp, num_simulations=5000, random_seed=42)
            
            risk_summary = pd.DataFrame({
                "Risk Model": ["Delta-Normal (Linear)", "Delta-Gamma (Cornish-Fisher)", "Monte Carlo (Full Reval)"],
                "Value at Risk (VaR)": [f"${dn_var:.2f}", f"${dg_var:.2f}", f"${mc_var:.2f}"],
                "Expected Shortfall (ES)": [f"${dn_es:.2f}", f"${dg_es:.2f}", f"${mc_es:.2f}"]
            })
            st.dataframe(risk_summary, use_container_width=True)
            
            st.subheader("Portfolio 2D Sensitivity Heatmap")
            spot_shifts = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
            vol_shifts = [-0.10, -0.05, 0.0, 0.05, 0.10]
            
            stress_df = port_engine.stress_test(spot_shifts, vol_shifts)
            
            fig_heat = px.imshow(
                stress_df,
                labels=dict(x="Underlying Spot Shift", y="Volatility Shift", color="Return %"),
                x=stress_df.columns,
                y=stress_df.index,
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0.0
            )
            fig_heat.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_heat, use_container_width=True)


# -------------------------------------------------------------
# TAB 10: Option Chain Analytics
# -------------------------------------------------------------
with tabs[9]:
    st.header("Option Chain Analytics (Live Market Data)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Query Live Chains")
        ticker_symbol = st.text_input("Enter Ticker Symbol", value="AAPL")
        
        # Download expirations
        try:
            chain_engine = OptionChainAnalytics(ticker_symbol, risk_free_rate=r_default, dividend_yield=div_default)
            expirations = chain_engine.get_expirations()
            
            if not expirations:
                st.error("No option chains found for this ticker.")
                active_exp = None
            else:
                active_exp = st.selectbox("Select Expiration Date", expirations)
        except Exception as e:
            st.error(f"Error fetching ticker info: {e}")
            active_exp = None
            
        # Filters
        st.subheader("Liquidity & Arbitrage Filters")
        min_vol_filter = st.slider("Minimum Volume", 0, 500, 0, 10)
        max_spread_pct_filter = st.slider("Maximum Bid-Ask Spread %", 0.0, 200.0, 100.0, 5.0)
        arb_filter = st.checkbox("Filter Arbitrage Violators", value=True)
            
        analyze_clicked = st.button("Load and Analyze Chain")
        
    with col2:
        if analyze_clicked and active_exp:
            with st.spinner("Downloading live options chain and calculating Greeks..."):
                try:
                    calls_raw, puts_raw = chain_engine.analyze_chain(active_exp)
                    spot_fetched = chain_engine.get_current_spot()
                    
                    # Compute time to maturity for arbitrage boundary checks
                    exp_dt = datetime.datetime.strptime(active_exp, "%Y-%m-%d").date()
                    days_to_mat = (exp_dt - datetime.date.today()).days
                    T = max(1, days_to_mat) / 365.0
                    
                    # Filter dataframes
                    calls_analyzed = chain_engine.filter_chain(
                        calls_raw, OptionType.CALL, spot_fetched, T,
                        max_spread_pct=max_spread_pct_filter, min_volume=min_vol_filter, filter_arbitrage=arb_filter
                    )
                    puts_analyzed = chain_engine.filter_chain(
                        puts_raw, OptionType.PUT, spot_fetched, T,
                        max_spread_pct=max_spread_pct_filter, min_volume=min_vol_filter, filter_arbitrage=arb_filter
                    )
                    
                    st.success(f"Loaded {ticker_symbol} chain for {active_exp}. Spot price: ${spot_fetched:.2f}")
                    
                    # Aggregate stats
                    oi_stats = chain_engine.get_open_interest_summary(calls_analyzed, puts_analyzed)
                    vol_stats = chain_engine.get_volume_summary(calls_analyzed, puts_analyzed)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    oi_ratio_val = oi_stats.get('put_call_oi_ratio')
                    vol_ratio_val = vol_stats.get('put_call_vol_ratio')
                    max_call_strike = oi_stats.get('max_call_oi_strike')
                    max_put_strike = oi_stats.get('max_put_oi_strike')
                    
                    oi_ratio_str = f"{oi_ratio_val:.3f}" if (oi_ratio_val is not None and not pd.isna(oi_ratio_val)) else "N/A"
                    vol_ratio_str = f"{vol_ratio_val:.3f}" if (vol_ratio_val is not None and not pd.isna(vol_ratio_val)) else "N/A"
                    max_call_strike_str = f"${max_call_strike:.2f}" if (max_call_strike is not None and not pd.isna(max_call_strike)) else "N/A"
                    max_put_strike_str = f"${max_put_strike:.2f}" if (max_put_strike is not None and not pd.isna(max_put_strike)) else "N/A"
                    
                    c1.metric("Put/Call OI Ratio", oi_ratio_str)
                    c2.metric("Put/Call Vol Ratio", vol_ratio_str)
                    c3.metric("Max Call OI Strike", max_call_strike_str)
                    c4.metric("Max Put OI Strike", max_put_strike_str)
                    
                    st.subheader("Options Chains Data")
                    opt_select = st.radio("Display Chain", ["Calls", "Puts"])
                    active_df = calls_analyzed if opt_select == "Calls" else puts_analyzed
                    
                    cols_to_show = [
                        "strike", "bid", "ask", "lastPrice", "mid_price", "volume", "openInterest",
                        "bid_iv", "ask_iv", "vol_spread", "computed_iv",
                        "delta", "gamma", "vega", "theta", "rho"
                    ]
                    
                    st.dataframe(
                        active_df[cols_to_show],
                        use_container_width=True
                    )
                    
                    # Plotly volume vs Open Interest
                    fig_chain = go.Figure()
                    fig_chain.add_trace(go.Bar(x=active_df["strike"], y=active_df["volume"], name="Volume", marker_color="#0099FF"))
                    fig_chain.add_trace(go.Bar(x=active_df["strike"], y=active_df["openInterest"], name="Open Interest", marker_color="#FF9900", opacity=0.7))
                    fig_chain.update_layout(
                        title="Volume vs. Open Interest by Strike",
                        barmode="group",
                        template="plotly_dark",
                        xaxis_title="Strike Price"
                    )
                    st.plotly_chart(fig_chain, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Failed to analyze options chain: {e}")
        else:
            st.info("Query and load option chains by clicking 'Load and Analyze Chain'.")


# -------------------------------------------------------------
# TAB 11: Volatility Forecasting ML
# -------------------------------------------------------------
with tabs[10]:
    st.header("Volatility Forecasting Framework")
    st.write(
        "Forecast future realized volatility using statistical baselines, econometric time series (EWMA, GARCH), "
        "multiscale autoregression (HAR-RV), and machine learning models, validated using rolling walk-forward validation."
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Model Configuration")
        ticker_fc = st.text_input("Forecast Ticker Symbol", value="AAPL", key="ticker_fc_val")
        forecast_horizon = st.slider("Forecast Horizon (Days)", 1, 10, 5, 1)
        split_ratio = st.slider("Train/Test Split Ratio", 0.50, 0.90, 0.70, 0.05)
        step_size = st.slider("Walk-Forward Re-calibration Step (Days)", 1, 21, 5, 1)
        
        train_model_btn = st.button("Run Volatility Forecasts")
        
    with col2:
        if train_model_btn:
            with st.spinner("Downloading historical data, engineering features, and running rolling walk-forward validation..."):
                try:
                    # Download 2 years of daily data
                    df_hist = yf.download(ticker_fc, period="2y")
                    if isinstance(df_hist.columns, pd.MultiIndex):
                        df_hist.columns = df_hist.columns.get_level_values(0)
                        
                    forecaster = VolatilityForecaster(df_hist)
                    # This engineers features and sets target_horizon
                    X, y = forecaster.engineer_features(target_horizon=forecast_horizon)
                    
                    results = forecaster.train_and_evaluate(train_size=split_ratio, step_size=step_size)
                    
                    # Display metrics table
                    metrics_rows = []
                    for model_name, res in results.items():
                        metrics_rows.append({
                            "Model": model_name,
                            "RMSE (%)": f"{res['rmse'] * 100:.3f}%",
                            "MAE (%)": f"{res['mae'] * 100:.3f}%",
                            "R² Score": f"{res['r2']:.4f}"
                        })
                    st.subheader("Rolling Walk-Forward Evaluation Metrics")
                    st.table(pd.DataFrame(metrics_rows))
                    
                    # Model selector for plotting
                    st.subheader("Interactive Forecast Overlay")
                    models_to_plot = st.multiselect(
                        "Overlay Models:",
                        options=list(results.keys()),
                        default=["Historical Baseline", "GARCH(1,1)", "HAR-RV Model", "Random Forest"]
                    )
                    
                    fig_fc = go.Figure()
                    
                    # Plot actual realized volatility (taken from the first available model)
                    first_model = list(results.keys())[0]
                    actuals = results[first_model]["actuals"]
                    fig_fc.add_trace(go.Scatter(
                        x=actuals.index, 
                        y=actuals.values * 100, 
                        name="Realized Volatility (Actual)", 
                        line=dict(color="white", width=2)
                    ))
                    
                    # Colors for models to keep it beautiful
                    colors = {
                        "Historical Baseline": "#888888",
                        "EWMA Volatility": "#FF9900",
                        "GARCH(1,1)": "#EF553B",
                        "HAR-RV Model": "#00CC96",
                        "Linear Regression": "#19D3F3",
                        "Random Forest": "#AB63FA",
                        "XGBoost": "#0099FF"
                    }
                    
                    for name in models_to_plot:
                        preds = results[name]["predictions"]
                        fig_fc.add_trace(go.Scatter(
                            x=preds.index,
                            y=preds.values * 100,
                            name=name,
                            line=dict(color=colors.get(name), width=1.5)
                        ))
                        
                    fig_fc.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Annualized Volatility (%)",
                        template="plotly_dark",
                        margin=dict(l=20, r=20, t=20, b=20),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)
                    
                    # Add detailed quantitative insights
                    st.subheader("Quantitative Insights & Model Interpretation")
                    st.markdown("""
                        ### 1. Parametric & Statistical Models (EWMA & GARCH)
                        * **GARCH(1,1)** models the mean-reverting behavior of volatility towards its long-term average ($V_{uncond}$) and captures **volatility clustering** (large shocks are followed by large shocks). It is highly robust because it has a strong econometric structure, preventing extreme out-of-sample prediction drift.
                        * **EWMA (RiskMetrics)** uses a fixed decay factor $\\lambda=0.94$. It acts as an exponentially decaying filter that reacts very quickly to new market shocks, but assumes volatility is a random walk with zero mean reversion, which can lead to lag during quiet periods.

                        ### 2. Autoregressive Scale Models (HAR-RV)
                        * The **HAR-RV Model** (Heterogeneous Autoregressive) regresses target realized volatility on daily, weekly, and monthly realized volatilities. This reflects the **Heterogeneous Market Hypothesis**, where short-term, medium-term, and long-term traders have different investment horizons and react to volatility differently.
                        * Despite its simplicity (linear OLS), HAR-RV is notoriously difficult to beat because it directly captures the multiscale cascading structure of volatility.

                        ### 3. Machine Learning Models (Linear Regression, RF, XGBoost)
                        * **Random Forest** and **XGBoost** can capture non-linear interactions between returns, daily ranges (Parkinson volatility), and higher-order moments (skewness/kurtosis).
                        * However, **machine learning models are prone to overfitting** on quiet regimes and can fail dramatically during sudden regime shifts (e.g., black swan events) since they cannot extrapolate beyond their training domain.
                        
                        ### 4. Historical Baseline
                        * The **Historical Baseline** (21-day realized volatility) represents the naive benchmark. Any forecasting methodology must outperform this baseline (i.e., achieve lower RMSE/MAE and higher $R^2$) to be considered useful.
                    """)
                    
                except Exception as e:
                    st.error(f"Error running forecasts: {e}")
        else:
            st.info("Query a ticker and run volatility forecasts to execute walk-forward validation and view comparative metrics.")
