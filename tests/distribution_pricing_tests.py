from csv import DictWriter
import importlib.util
from math import isclose
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bucket_schema = load_module_from_path("bucket_schema", SRC_DIR / "bucket_schema.py")
error_boundaries = load_module_from_path("error_boundaries", SRC_DIR / "error_boundaries.py")
distribution_pricing = load_module_from_path(
    "distribution_pricing", SRC_DIR / "distribution_pricing.py"
)

convert_market_to_boundaries = error_boundaries.convert_market_to_boundaries
normal_bucket_prob = distribution_pricing.normal_bucket_prob


def generate_bucket_probability_demo(output_file: Path) -> list[dict[str, float | str]]:
    mu = 92
    sigma = 2
    buckets = convert_market_to_boundaries(
        [88, 89, 90, 91, 92, 93, 94, 95, 96, 97],
        "New York",
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    with output_file.open("w", newline="") as handle:
        writer = DictWriter(
            handle,
            fieldnames=["bucket_label", "lower_bound", "upper_bound", "probability"],
            lineterminator="\n",
        )
        writer.writeheader()
        for bucket in buckets:
            probability = normal_bucket_prob(bucket.lower_bound, bucket.upper_bound, mu, sigma)
            row = {
                "bucket_label": bucket.name,
                "lower_bound": "-inf" if bucket.lower_bound is None else bucket.lower_bound,
                "upper_bound": "inf" if bucket.upper_bound is None else bucket.upper_bound,
                "probability": float(probability),
            }
            writer.writerow(row)
            rows.append(row)

    return rows


def main() -> None:
    output_file = ROOT / "outputs" / "bucket_probability_demo.csv"
    rows = generate_bucket_probability_demo(output_file)

    assert all(isinstance(row["probability"], float) for row in rows), "Probabilities must be numeric"
    assert all(0.0 <= row["probability"] <= 1.0 for row in rows), "Probabilities must be between 0 and 1"
    assert [row["bucket_label"] for row in rows] == [
        "88 or lower",
        "89 to 90",
        "91 to 92",
        "93 to 94",
        "95 to 96",
        "97 or higher",
    ], "Buckets should follow Kalshi two-degree interior structure"
    assert isclose(
        sum(float(row["probability"]) for row in rows),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ), "Exhaustive bucket probabilities must sum to 1"

    print(f"Saved {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    main()
