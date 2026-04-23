import numpy as np
from itertools import combinations
ALPHA = 0.1 # 90% CI
def get_multiplier(len_runs: int, t: float = 0.95):
    if t == 0.95:
        if len_runs == 3:
            return 2.9
        elif len_runs == 6:
            return 2.015
        else:
            return 1.833
    elif t == 0.90:
        if len_runs == 3:
            return 1.886
        elif len_runs == 6:
            return 1.476
        else:
            return 1.383
    elif t == 0.85:
        if len_runs == 3:
            return 1.386
        elif len_runs == 6:
            return 1.156
        else:
            return 1.1
    elif t == 0.80:
        if len_runs == 3:
            return 1.061
        elif len_runs == 6:
            return 0.920
        else:
            return 0.883
    else:
        return 0.0


def bootstrap_ci_mean(x, B=5000, alpha=ALPHA, random_state=None, max_num_sims=None):
    """
    Compute a bootstrap confidence interval for the mean of the input samples.

    Parameters
    ----------
    x : array-like
        Original data (e.g., [x1, x2, ..., x10]).
    B : int, optional
        Number of bootstrap replicates (default=1000).
    alpha : float, optional
        Significance level (default=0.05 for 95% CI).
    random_state : int or None
        Random seed for reproducibility.
    max_num_sims : int or None, optional
        Maximum number of samples to use for bootstrap (default=None uses all).

    Returns
    -------
    point_est : float
        Sample mean of the observed data.
    ci_lower : float
        Lower bootstrap percentile bound.
    ci_upper : float
        Upper bootstrap percentile bound.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    
    if max_num_sims is None:
        max_num_sims = len(x)
    else:
        max_num_sims = min(max_num_sims, len(x))

    # Point estimate: mean of observed data
    point_est = np.mean(x)

    # Vectorized bootstrap replicates - much faster!
    # Generate all bootstrap samples at once
    bootstrap_indices = rng.choice(len(x), size=(B, max_num_sims), replace=True)
    bootstrap_samples = x[bootstrap_indices]  # Shape: (B, max_num_sims)
    boot_means = np.mean(bootstrap_samples, axis=1)  # Shape: (B,)

    # Percentile CI
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return point_est, lower, upper


def bootstrap_ci_best_of_n(x, n, B=5000, alpha=ALPHA, random_state=None, max_num_sims=None, higher_is_better=True):
    """
    Compute a bootstrap confidence interval for the mean of the "best-of-n" statistic:
    the expected best of n draws from the underlying distribution.

    Parameters
    ----------
    x : array-like
        Original data (e.g., [x1, x2, ..., x10]). Must be in "higher is better" form
        if higher_is_better=True, or "lower is better" form if higher_is_better=False.
    n : int
        Number of samples in each "best-of-n" draw (e.g., n=4 for best of 4).
    B : int, optional
        Number of bootstrap replicates (default=5000).
    alpha : float, optional
        Significance level (default=0.1 for 90% CI).
    random_state : int or None
        Random seed for reproducibility.
    max_num_sims : int, optional
        Number of Monte Carlo simulations per bootstrap replicate to approximate
        the expected best (default=None uses len(x)).
    higher_is_better : bool, optional
        If True, "best" = maximum (e.g. accuracy, reward, negative cost).
        If False, "best" = minimum (e.g. cost, latency). Default True.

    Returns
    -------
    point_est : float
        Point estimate: average of all observed n-combination best values.
    ci_lower : float
        Lower bootstrap percentile bound.
    ci_upper : float
        Upper bootstrap percentile bound.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    if max_num_sims is None:
        max_num_sims = len(x)
    else:
        max_num_sims = min(max_num_sims, len(x))

    best_fn = np.max if higher_is_better else np.min

    # ---- Point estimate using all combinations (if small enough) ----
    if len(x) <= 20 and n <= len(x):  # brute force all combinations if feasible
        best_vals = [best_fn(x[list(idx)]) for idx in combinations(range(len(x)), n)]
        point_est = np.mean(best_vals)
    else:
        # fallback: approximate with Monte Carlo
        sim_indices = rng.choice(len(x), size=(max_num_sims, n), replace=True)
        best_vals = best_fn(x[sim_indices], axis=1)
        point_est = np.mean(best_vals)
    # ---- Bootstrap replicates ----
    boot_stats = []
    for _ in range(B):
        x_boot = rng.choice(x, size=len(x), replace=True)
        sim_indices = rng.choice(len(x_boot), size=(max_num_sims, n), replace=True)
        best_boot = best_fn(x_boot[sim_indices], axis=1)
        boot_stats.append(np.mean(best_boot))

    boot_stats = np.array(boot_stats)

    # Percentile CI
    lower = np.percentile(boot_stats, 100 * (alpha / 2))
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return point_est, lower, upper