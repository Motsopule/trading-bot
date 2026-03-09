"""
Monte Carlo analysis of backtest trade logs.

Estimates losing streak probability and drawdown distribution by running
many simulations with randomly shuffled trade order.
"""

import glob
import os
import random
from typing import List, Tuple

import pandas as pd


def load_trade_returns(csv_pattern: str = "backtest_trades_*.csv") -> List[float]:
    """
    Load all backtest trade CSVs matching the pattern and extract return_pct.

    Args:
        csv_pattern: Glob pattern for backtest trade CSV files.

    Returns:
        List of per-trade returns in decimal form (return_pct / 100).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_path = os.path.join(script_dir, csv_pattern)
    files = sorted(glob.glob(search_path))
    if not files:
        raise FileNotFoundError(f"No files matching {csv_pattern} in {script_dir}")

    all_returns = []
    for path in files:
        df = pd.read_csv(path)
        if "return_pct" not in df.columns:
            raise ValueError(f"Missing column 'return_pct' in {path}")
        # Convert percentage points to decimal (e.g. 1.5 -> 0.015)
        all_returns.extend((df["return_pct"] / 100.0).tolist())

    return all_returns


def max_losing_streak(returns: List[float]) -> int:
    """
    Compute the maximum number of consecutive losing trades.

    Args:
        returns: List of per-trade decimal returns.

    Returns:
        Length of the longest run of trades with negative return.
    """
    if not returns:
        return 0
    best = 0
    current = 0
    for r in returns:
        if r < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def max_drawdown(returns: List[float]) -> float:
    """
    Compute maximum drawdown from a sequence of period returns.

    Uses cumulative wealth (start 1.0, multiply by (1 + r)) and measures
    the largest peak-to-trough decline as a fraction of peak.

    Args:
        returns: List of per-trade decimal returns.

    Returns:
        Maximum drawdown as a decimal (e.g. 0.15 = 15%).
    """
    if not returns:
        return 0.0
    peak = 1.0
    max_dd = 0.0
    wealth = 1.0
    for r in returns:
        wealth *= 1.0 + r
        peak = max(peak, wealth)
        dd = (peak - wealth) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def final_return(returns: List[float]) -> float:
    """
    Compute total (compounded) return over the sequence.

    Args:
        returns: List of per-trade decimal returns.

    Returns:
        Final cumulative return as decimal (e.g. 0.20 = 20%).
    """
    if not returns:
        return 0.0
    wealth = 1.0
    for r in returns:
        wealth *= 1.0 + r
    return wealth - 1.0


def run_simulation(returns: List[float]) -> Tuple[int, float, float]:
    """
    Run one Monte Carlo trial: shuffle returns and compute metrics.

    Args:
        returns: List of per-trade decimal returns (will be shuffled in place).

    Returns:
        (max_losing_streak, max_drawdown, final_return).
    """
    shuffled = returns.copy()
    random.shuffle(shuffled)
    streak = max_losing_streak(shuffled)
    dd = max_drawdown(shuffled)
    total_ret = final_return(shuffled)
    return streak, dd, total_ret


def run_monte_carlo(
    returns: List[float], n_simulations: int = 10_000, seed: int = 42
) -> pd.DataFrame:
    """
    Run multiple Monte Carlo simulations and collect results.

    Args:
        returns: List of per-trade decimal returns.
        n_simulations: Number of simulations to run.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: max_losing_streak, max_drawdown, final_return.
    """
    random.seed(seed)
    results = []
    for _ in range(n_simulations):
        streak, dd, total_ret = run_simulation(returns)
        results.append(
            {
                "max_losing_streak": streak,
                "max_drawdown": dd,
                "final_return": total_ret,
            }
        )
    return pd.DataFrame(results)


def main() -> None:
    """Load trades, run Monte Carlo, print summary, and save results."""
    n_simulations = 10_000
    print("Loading backtest trade logs...")
    returns = load_trade_returns()
    print(f"Loaded {len(returns)} trades from backtest_trades_*.csv")

    print(f"Running {n_simulations} Monte Carlo simulations...")
    df = run_monte_carlo(returns, n_simulations=n_simulations)

    # Summary statistics
    streak = df["max_losing_streak"]
    dd = df["max_drawdown"]

    expected_streak = float(streak.mean())
    worst_streak_95 = int(streak.quantile(0.95))
    worst_streak_99 = int(streak.quantile(0.99))
    avg_drawdown = float(dd.mean())
    worst_drawdown = float(dd.max())

    print("\n" + "=" * 50)
    print("MONTE CARLO ANALYSIS RESULTS")
    print("=" * 50)
    print(f"  Expected max losing streak:     {expected_streak:.2f}")
    print(f"  95% worst-case losing streak:   {worst_streak_95}")
    print(f"  99% worst-case losing streak:   {worst_streak_99}")
    print(f"  Average max drawdown:           {avg_drawdown:.2%}")
    print(f"  Worst max drawdown:             {worst_drawdown:.2%}")
    print("=" * 50)

    # Save summary to CSV (one row of key metrics plus optional full sim data)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, "monte_carlo_results.csv")
    summary_df = pd.DataFrame(
        [
            {
                "metric": "expected_max_losing_streak",
                "value": expected_streak,
            },
            {
                "metric": "worst_case_95_losing_streak",
                "value": worst_streak_95,
            },
            {
                "metric": "worst_case_99_losing_streak",
                "value": worst_streak_99,
            },
            {
                "metric": "average_drawdown",
                "value": avg_drawdown,
            },
            {
                "metric": "worst_drawdown",
                "value": worst_drawdown,
            },
        ]
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
