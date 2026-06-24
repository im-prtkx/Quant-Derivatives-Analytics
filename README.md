# Quant Derivatives Analytics Platform

A production-quality derivatives pricing, risk management, and volatility analytics platform in Python.

This project implements advanced financial modeling engines, interactive risk monitoring tools, machine learning forecasting, and an interactive Streamlit dashboard.

---

## Architecture and Project Structure

```
derivatives_platform/
├── src/
│   ├── pricing/
│   │   ├── __init__.py
│   │   ├── black_scholes.py       # Black-Scholes pricing model
│   │   ├── binomial_tree.py       # Cox-Ross-Rubinstein & Leisen-Reimer tree models
│   │   └── monte_carlo.py         # Monte Carlo path simulation pricing (GBM & Heston, CV)
│   ├── greeks/
│   │   ├── __init__.py
│   │   └── greeks.py              # Analytical Greeks & Higher-Order Greeks solver
│   ├── volatility/
│   │   ├── __init__.py
│   │   ├── implied_vol.py         # Newton-Raphson, Halley & Bisection IV solver
│   │   ├── historical_vol.py      # Rolling vol & regime detection
│   │   ├── vol_smile.py           # Smile analytics & SABR Calibration
│   │   ├── vol_surface.py         # 3D surface modeling & interpolation
│   │   └── vol_forecasting.py     # Vol forecasting with ML (XGBoost, RF)
│   ├── strategies/
│   │   ├── __init__.py
│   │   └── strategy.py            # Payoff & metric builder for 9 option strategies
│   ├── portfolio/
│   │   ├── __init__.py
│   │   └── portfolio.py           # Greeks aggregator, 2D stress testing & VaR/ES (Cornish-Fisher)
│   ├── option_chain/
│   │   ├── __init__.py
│   │   └── chain_analytics.py     # yfinance live options chain analyst (Filters, Bid-Ask IV)
│   ├── backtesting/
│   │   ├── __init__.py
│   │   └── backtester.py          # CAGR, Sharpe, Drawdown backtesting engine
│   └── utils/
├── dashboard/
│   └── app.py                     # Streamlit dashboard
├── tests/                         # Pytest test cases & benchmark.py script
├── report/
│   └── project_report.md          # Technical quantitative project report
├── README.md                      # General documentation
└── requirements.txt
```

---

## Installation

Ensure you have Python 3.13+ installed.

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Quant-Derivatives-Analytics
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas scipy plotly matplotlib yfinance streamlit scikit-learn xgboost pytest
   ```

---

## Usage

### Running Tests
To run the automated test suite of 65 test cases:
```bash
pytest
```

### Running Performance Benchmarks
To run the numerical pricing and solver benchmarking suite:
```bash
PYTHONPATH=. python tests/benchmark.py
```

### Starting the Dashboard
To start the interactive Streamlit analytics platform:
```bash
streamlit run dashboard/app.py
```

---

## Mathematical Foundations

### 1. Black-Scholes-Merton (1973) & Greeks Engine
Prices European Call ($C$) and Put ($P$) options on dividend-paying assets:
$$C = S e^{-q T} N(d_1) - K e^{-r T} N(d_2)$$
$$P = K e^{-r T} N(-d_2) - S e^{-q T} N(-d_1)$$
where:
$$d_1 = \frac{\ln(S/K) + (r - q + 0.5 \sigma^2)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}$$

The Greeks engine computes first-order, second-order, and higher-order sensitivities analytically:
- **Vanna** ($\frac{\partial \Delta}{\partial \sigma}$): Sensitivity of Delta to volatility.
- **Volga / Vomma** ($\frac{\partial^2 V}{\partial \sigma^2}$): Sensitivity of Vega to volatility.
- **Charm** ($\frac{\partial \Delta}{\partial T}$): Delta decay over time.
- **Speed** ($\frac{\partial \Gamma}{\partial S}$): Change of Gamma with respect to Spot.
- **Color** ($\frac{\partial \Gamma}{\partial T}$): Gamma decay over time.

### 2. Leisen-Reimer Trees & Early Exercise Boundary
Leisen-Reimer trees remove the convergence "sawtooth" oscillations seen in Cox-Ross-Rubinstein trees by using the Peizer-Pratt inversion method to align the median of the binomial tree node distribution with the strike price $K$ at maturity.
American option valuation incorporates early exercise boundary tracking to identify the critical spot prices $S^*(t)$ where immediate exercise becomes optimal.

### 3. Monte Carlo & Stochastic Volatility
- **Control Variates**: European option Monte Carlo pricing utilizes the underlying spot price $S_T$ as a control variate to achieve significant variance reduction:
  $$Y_{CV} = Y - \beta (S_T - E[S_T])$$
  where $\beta = \text{Cov}(Y, S_T) / \text{Var}(S_T)$ and $E[S_T] = S_0 e^{(r-q)T}$.
- **Heston Stochastic Volatility**: Simulates joint stock price and stochastic variance trajectories:
  $$dS_t = (r - q) S_t dt + \sqrt{v_t} S_t dW_t^S$$
  $$dv_t = \kappa(\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v$$
  with correlation $d\langle W^S, W^v \rangle_t = \rho dt$.

### 4. Volatility smile (SABR Calibration)
- **Halley's Method**: Implements a third-order root-finding method for implied volatility:
  $$\sigma_{n+1} = \sigma_n - \frac{2 (BS(\sigma_n) - P_{mkt}) \text{Vega}(\sigma_n)}{2 \text{Vega}(\sigma_n)^2 - (BS(\sigma_n) - P_{mkt}) \text{Volga}(\sigma_n)}$$
- **SABR Model**: Fits Hagan's volatility smile formula to market options data to calibrate the initial volatility ($\alpha$), correlation ($\rho$), and vol-of-vol ($\nu$) parameters.

### 5. Portfolio Risk (VaR & Expected Shortfall)
Calculates Value at Risk (VaR) and Expected Shortfall (ES) at a given confidence level $\alpha$ for holding period $H$:
- **Delta-Normal**: Linear risk approximation assuming normal asset returns.
- **Delta-Gamma (Cornish-Fisher)**: Quadratic risk approximation using the Cornish-Fisher expansion to adjust the normal quantiles for skewness and excess kurtosis.
- **Monte Carlo**: Non-linear full-revaluation simulation of portfolio returns.
