from csv import DictWriter
import importlib.util
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


distribution_pricing = load_module_from_path(
    "distribution_pricing", SRC_DIR / "distribution_pricing.py"
)
bucket_schema = load_module_from_path("bucket_schema", SRC_DIR / "bucket_schema.py")

Bucket = bucket_schema.Bucket
normal_bucket_prob = distribution_pricing.normal_bucket_prob


def generate_bucket_probability_demo(output_file: Path) -> list[dict[str, float | str]]:
    mu = 92
    sigma = 2
    buckets = [
        Bucket(location="New York", left_bound=None, right_bound=88),
        Bucket(location="New York", left_bound=89, right_bound=90),
        Bucket(location="New York", left_bound=91, right_bound=92),
        Bucket(location="New York", left_bound=93, right_bound=94),
        Bucket(location="New York", left_bound=95, right_bound=96),
        Bucket(location="New York", left_bound=97, right_bound=None)
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    with output_file.open("w", newline="") as handle:
        writer = DictWriter(
            handle,
            fieldnames=["bucket_label", "lower_bound", "upper_bound", "probability"],
        )
        writer.writeheader()
        for bucket in buckets:
            probability = normal_bucket_prob(bucket.left_bound, bucket.right_bound, mu, sigma)
            row = {
                "bucket_label": bucket.location,
                "lower_bound": bucket.left_bound,
                "upper_bound": bucket.right_bound,
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

    print(f"Saved {len(rows)} rows to {output_file}")


if __name__ == "__main__":
    main()
