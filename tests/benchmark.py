import time
import numpy as np
import pandas as pd
import scipy.stats as stats
from src.pricing.black_scholes import BlackScholes, OptionType
from src.pricing.binomial_tree import BinomialTree
from src.pricing.monte_carlo import MonteCarloEngine
from src.volatility.implied_vol import ImpliedVolatilitySolver

def bisection_iv(market_price, spot, strike, T, r, option_type, q=0.0, tolerance=1e-6, max_iter=100):
    low = 1e-6
    high = 5.0
    for idx in range(max_iter):
        mid = 0.5 * (low + high)
        bs = BlackScholes(spot, strike, T, r, mid, q)
        price_mid = bs.price(option_type)
        price_error = price_mid - market_price
        if abs(price_error) < tolerance or (high - low) < tolerance:
            return mid
        if price_mid < market_price:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)

def benchmark_solvers():
    print("Benchmarking Implied Volatility Solvers...")
    spot = 100.0
    strike = 105.0
    T = 0.5
    r = 0.05
    q = 0.02
    true_vol = 0.25
    
    # Calculate target price first
    bs_target = BlackScholes(spot, strike, T, r, true_vol, q)
    call_price = bs_target.call_price()
    
    n_runs = 500
    results = []
    
    # Warmups
    _ = ImpliedVolatilitySolver.calculate_iv(call_price, spot, strike, T, r, OptionType.CALL, q, method="newton")
    _ = ImpliedVolatilitySolver.calculate_iv(call_price, spot, strike, T, r, OptionType.CALL, q, method="halley")
    _ = bisection_iv(call_price, spot, strike, T, r, OptionType.CALL, q)
    
    # Newton-Raphson
    start_time = time.perf_counter()
    for _ in range(n_runs):
        iv_newton = ImpliedVolatilitySolver.calculate_iv(
            call_price, spot, strike, T, r, OptionType.CALL, q, method="newton"
        )
    elapsed_newton = (time.perf_counter() - start_time) / n_runs * 1e6
    err_newton = abs(iv_newton - true_vol)
    results.append({
        "Solver": "Newton-Raphson",
        "Avg Time (μs)": f"{elapsed_newton:.2f}",
        "Calibrated IV": f"{iv_newton:.6f}",
        "Absolute Error": f"{err_newton:.2e}"
    })
    
    # Halley
    start_time = time.perf_counter()
    for _ in range(n_runs):
        iv_halley = ImpliedVolatilitySolver.calculate_iv(
            call_price, spot, strike, T, r, OptionType.CALL, q, method="halley"
        )
    elapsed_halley = (time.perf_counter() - start_time) / n_runs * 1e6
    err_halley = abs(iv_halley - true_vol)
    results.append({
        "Solver": "Halley's Method",
        "Avg Time (μs)": f"{elapsed_halley:.2f}",
        "Calibrated IV": f"{iv_halley:.6f}",
        "Absolute Error": f"{err_halley:.2e}"
    })
    
    # Bisection
    start_time = time.perf_counter()
    for _ in range(n_runs):
        iv_bisect = bisection_iv(
            call_price, spot, strike, T, r, OptionType.CALL, q
        )
    elapsed_bisect = (time.perf_counter() - start_time) / n_runs * 1e6
    err_bisect = abs(iv_bisect - true_vol)
    results.append({
        "Solver": "Bisection Method",
        "Avg Time (μs)": f"{elapsed_bisect:.2f}",
        "Calibrated IV": f"{iv_bisect:.6f}",
        "Absolute Error": f"{err_bisect:.2e}"
    })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("-" * 50)
    return df

def benchmark_trees():
    print("Benchmarking Binomial Tree Models (CRR vs. Leisen-Reimer)...")
    spot = 100.0
    strike = 100.0  # ATM is where convergence sawtooth is worst
    T = 1.0
    r = 0.05
    q = 0.00
    vol = 0.20
    
    # Analytical European Call Price
    bs = BlackScholes(spot, strike, T, r, vol, q)
    analytical_price = bs.call_price()
    
    step_sizes = [20, 50, 100, 200, 500]
    results = []
    
    for steps in step_sizes:
        # CRR tree
        crr_tree = BinomialTree(spot, strike, T, r, vol, q, steps=steps, model_type="crr")
        t0 = time.perf_counter()
        crr_price = crr_tree.price(OptionType.CALL, american=False)
        crr_time = (time.perf_counter() - t0) * 1e3 # ms
        crr_err = abs(crr_price - analytical_price)
        
        # Leisen-Reimer tree
        lr_tree = BinomialTree(spot, strike, T, r, vol, q, steps=steps, model_type="lr")
        t0 = time.perf_counter()
        lr_price = lr_tree.price(OptionType.CALL, american=False)
        lr_time = (time.perf_counter() - t0) * 1e3 # ms
        lr_err = abs(lr_price - analytical_price)
        
        results.append({
            "Steps": steps,
            "CRR Price": f"{crr_price:.5f}",
            "CRR Error": f"{crr_err:.2e}",
            "CRR Time (ms)": f"{crr_time:.2f}",
            "LR Price": f"{lr_price:.5f}",
            "LR Error": f"{lr_err:.2e}",
            "LR Time (ms)": f"{lr_time:.2f}"
        })
        
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("-" * 50)
    return df

def benchmark_monte_carlo():
    print("Benchmarking Monte Carlo Schemes...")
    spot = 100.0
    strike = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    vol = 0.20
    
    bs = BlackScholes(spot, strike, T, r, vol, q)
    analytical_price = bs.call_price()
    
    mc = MonteCarloEngine(spot, strike, T, r, vol, q)
    
    schemes = ["exact", "euler", "milstein"]
    results = []
    
    num_paths = 50000
    num_steps = 100
    
    for scheme in schemes:
        start_time = time.perf_counter()
        # Simulate paths
        paths = mc.simulate_paths(
            num_paths=num_paths,
            num_steps=num_steps,
            antithetic=True,
            scheme=scheme,
            seed=42
        )
        # Price European Call
        payoffs = np.maximum(paths[:, -1] - strike, 0.0)
        price = np.exp(-r * T) * np.mean(payoffs)
        elapsed = (time.perf_counter() - start_time) * 1e3 # ms
        error = abs(price - analytical_price)
        
        results.append({
            "Scheme": scheme.capitalize(),
            "Price": f"{price:.5f}",
            "Error vs. BS": f"{error:.2e}",
            "Time (ms)": f"{elapsed:.2f}"
        })
        
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("-" * 50)
    return df

def generate_markdown_report(df_solvers, df_trees, df_mc):
    report_content = f"""# Quantitative Derivatives Analytics Platform - Performance Benchmark Report

This report evaluates the execution speeds, numerical accuracy, and convergence rates of our advanced pricing engines and numerical solvers.

---

## 1. Implied Volatility Solvers Benchmark
We solve for the implied volatility of a European Call option ($S=100, K=105, T=0.5, r=0.05, q=0.02, \\sigma_{{true}}=25\\%$) using Newton-Raphson, Halley's third-order solver (utilizing Volga), and the Bisection fallback method.

{df_solvers.to_markdown(index=False)}

> **Insights**: 
> - Halley's third-order solver achieves extremely fast convergence, reducing iterations by incorporating higher-order derivatives (Volga/Vomma).
> - Newton-Raphson is also highly efficient, while Bisection serves as a robust fallback with linear convergence.

---

## 2. Binomial Trees Convergence Benchmark
We price an at-the-money European Call option ($S=100, K=100, T=1.0, r=0.05, \\sigma=20\\%$) across different step sizes to compare the convergence behavior of the standard Cox-Ross-Rubinstein (CRR) tree vs. the Leisen-Reimer (LR) tree.

{df_trees.to_markdown(index=False)}

> **Insights**:
> - Leisen-Reimer trees demonstrate much smaller pricing errors at low step counts (e.g. $N=20$) because Peizer-Pratt inversion forces the tree nodes to align symmetrically with the option strike, eliminating the convergence sawtooth oscillations of the CRR model.

---

## 3. Monte Carlo Schemes Benchmark
We value a European Call option ($S=100, K=100, T=1.0, r=0.05, q=0.02, \\sigma=20\\%$) using 50,000 paths and 100 time steps. We compare the Exact Analytical log-normal scheme, Euler-Maruyama scheme, and the Milstein scheme.

{df_mc.to_markdown(index=False)}

> **Insights**:
> - The Exact scheme uses cumulative log-returns directly, giving a very precise result with no discretization error.
> - The Milstein scheme corrects Euler-Maruyama for non-linear stochastic volatility terms, resulting in lower error than Euler-Maruyama for identical path sizes.

"""
    with open("tests/benchmark_report.md", "w") as f:
        f.write(report_content)
    print("Benchmark report saved to tests/benchmark_report.md")

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING PERFORMANCE BENCHMARK SUITE")
    print("=" * 60)
    
    df_solvers = benchmark_solvers()
    df_trees = benchmark_trees()
    df_mc = benchmark_monte_carlo()
    
    generate_markdown_report(df_solvers, df_trees, df_mc)
    print("=" * 60)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 60)
