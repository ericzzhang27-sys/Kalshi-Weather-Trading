from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

try:
    from .bucket_schema import Bucket
except ImportError:
    from bucket_schema import Bucket


def normal_cdf(x: float | np.ndarray, mu: float, sigma: float) -> float | np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma must be greater than 0")

    return norm.cdf(x, loc=mu, scale=sigma)

def plot_and_save_cdf(mu: float = 1, sigma: float = 1, filename: str = 'normal_cdf.png') -> None:
    """Plot normal CDF and save to figures folder."""
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100) 
    plt.figure(figsize=(10, 6))
    plt.plot(x, normal_cdf(x, mu, sigma))
    plt.title(f'Normal CDF (μ={mu}, σ={sigma})')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.grid(True, alpha=0.3)
    
    output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")



def normal_bucket_prob(lower_bound: float | None, upper_bound: float | None, mu: float, sigma: float) -> float:
    if lower_bound is None and upper_bound is None:
        raise ValueError("Both lower and upper bounds cannot be None")

    if lower_bound is not None and upper_bound is not None and upper_bound <= lower_bound:
        raise ValueError("upper_bound must be greater than lower_bound")

    if lower_bound is None:
        lower_bound = -np.inf
    if upper_bound is None:
        upper_bound = np.inf

    return float(normal_cdf(upper_bound, mu, sigma) - normal_cdf(lower_bound, mu, sigma))


def normal_bucket_probs(buckets: list[Bucket], mu: float, sigma: float) -> dict[str, float]:
    probability_list: dict[str, float] = {}
    for bucket in buckets:
        probability_list[bucket.name] = normal_bucket_prob(
            bucket.lower_bound,
            bucket.upper_bound,
            mu,
            sigma,
        )
    return probability_list



