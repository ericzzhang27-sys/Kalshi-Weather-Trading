import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from .bucket_schema import Bucket
    from .distribution_pricing import normal_cdf, normal_bucket_prob
    from .error_boundaries import convert_market_to_boundaries
except ImportError:
    from bucket_schema import Bucket
    from distribution_pricing import normal_cdf, normal_bucket_prob
    from error_boundaries import convert_market_to_boundaries

def get_bounds(buckets: list[Bucket]) -> list[float]:
    """Return sorted unique finite boundary values from buckets."""
    bounds = []
    for bucket in buckets:
        if bucket.lower_bound is not None:
            bounds.append(bucket.lower_bound)
        if bucket.upper_bound is not None:
            bounds.append(bucket.upper_bound)

    return sorted(set(bounds))


def plot_cdf_with_probabilities(
    buckets: list[Bucket],
    mu: float,
    sigma: float,
    filename: str = 'normal_cdf_shaded_probs.png',
) -> None:
    """Shade each bucket region (including unbounded) and add a legend with probabilities.

    Uses `normal_bucket_prob` from `distribution_pricing` to compute the probability
    for each region and maps a unique color to each region in the legend.
    """
    # plotting range
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 800)
    y = normal_cdf(x, mu, sigma)

    # derive bounds from helper and build contiguous regions including unbounded ends
    bounds = get_bounds(buckets)
    regions = []
    # left unbounded region
    if len(bounds) == 0:
        regions.append((None, None))
    else:
        regions.append((None, bounds[0]))
        for i in range(len(bounds) - 1):
            regions.append((bounds[i], bounds[i + 1]))
        regions.append((bounds[-1], None))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, color='black', linewidth=2, label='Normal CDF')

    cmap = plt.get_cmap('tab10')
    handles = []
    from matplotlib.patches import Patch

    for idx, (lower, upper) in enumerate(regions):
        color = cmap(idx % 10)

        # compute probability for the region using provided helper
        prob = normal_bucket_prob(lower, upper, mu, sigma)

        # determine mask for plotting (handle None bounds as unbounded)
        if lower is None and upper is None:
            mask = np.full_like(x, True, dtype=bool)
        elif lower is None:
            mask = x <= upper
        elif upper is None:
            mask = x >= lower
        else:
            mask = (x >= lower) & (x <= upper)

        if mask.any():
            ax.fill_between(x[mask], 0, y[mask], color=color, alpha=0.25)

        # format bounds for display
        display_lower = "-inf" if lower is None else f"{lower:.1f}"
        display_upper = "inf" if upper is None else f"{upper:.1f}"
        # create legend handle with bounds and probability formatted
        label_text = f"({display_lower}, {display_upper}) — p={prob:.4f}"
        handles.append(Patch(facecolor=color, edgecolor=color, label=label_text, alpha=0.6))

    ax.set_title(f'Normal CDF shaded by bucket regions (μ={mu}, σ={sigma})')
    ax.set_xlabel('x')
    ax.set_ylabel('CDF')
    ax.grid(True, alpha=0.3)
    ax.legend(handles=handles, loc='upper left', fontsize=8)

    output_dir = Path(__file__).parent.parent / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    market_boundaries = convert_market_to_boundaries([69, 70, 71, 72, 73, 74, 75, 76, 77, 78], "Chicago")
    bounds = get_bounds(market_boundaries)
    plot_cdf_with_probabilities(market_boundaries, 73.0, 1.8)
