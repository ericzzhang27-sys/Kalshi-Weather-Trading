from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hurdle_model import load_hurdle_predictor


MODELS_DIR = REPO_ROOT / "models"
STUDY_DIR = REPO_ROOT / "outputs" / "hurdle" / "challenger_study"
CANDIDATE_DIR = STUDY_DIR / "candidates"
DATASET_PATH = REPO_ROOT / "data" / "processed" / "hurdle_dataset.csv"
CLASSIFIER_PATH = MODELS_DIR / "hurdle_classifier.pkl"
CALIBRATOR_PATH = MODELS_DIR / "hurdle_calibrator.pkl"
FEATURES_PATH = MODELS_DIR / "hurdle_features.json"
METADATA_PATH = MODELS_DIR / "hurdle_metadata.json"
BUNDLE_PATH = MODELS_DIR / "exceedance_model_bundle.json"
REMAINING_METADATA_PATH = MODELS_DIR / "remaining_increase_metadata.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a validated hurdle challenger")
    parser.add_argument(
        "candidate",
        choices=["ngboost_bernoulli", "logistic_regression", "lightgbm_classifier"],
    )
    parser.add_argument(
        "--user-override",
        action="store_true",
        help="record an explicit user override when the candidate did not clear the automatic threshold",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = pd.read_csv(STUDY_DIR / "overall_test_comparison.csv")
    if args.candidate not in set(comparison["model"]):
        raise ValueError(f"Candidate {args.candidate!r} is absent from the completed study")
    candidate_row = comparison.loc[comparison["model"].eq(args.candidate)].iloc[0].to_dict()
    candidate_classifier = CANDIDATE_DIR / f"{args.candidate}_classifier.pkl"
    candidate_calibrator = CANDIDATE_DIR / f"{args.candidate}_calibrator.pkl"
    if not candidate_classifier.exists() or not candidate_calibrator.exists():
        raise FileNotFoundError("Validated candidate artifacts are missing")

    existing_bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    automatic_decision = existing_bundle.get("replacement_decision", {})
    automatically_selected = existing_bundle.get("winner") == args.candidate
    if not automatically_selected and not args.user_override:
        raise ValueError("Candidate did not win automatically; pass --user-override to record explicit authority")

    # Binary artifact promotion is a direct copy from the immutable study output.
    shutil.copy2(candidate_classifier, CLASSIFIER_PATH)
    shutil.copy2(candidate_calibrator, CALIBRATOR_PATH)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_status": "frozen_user_override_after_challenger_study"
            if args.user_override
            else "frozen_after_challenger_study",
            "model_type": args.candidate,
            "calibrator_type": str(candidate_row["calibration"]),
            "test_metrics": candidate_row,
            "promotion": {
                "candidate": args.candidate,
                "user_override": bool(args.user_override),
                "automatic_decision_preserved": automatic_decision,
                "reason": "explicit user instruction to replace incumbent"
                if args.user_override
                else "automatic challenger-study winner",
                "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_classifier_sha256": _sha256(candidate_classifier),
                "source_calibrator_sha256": _sha256(candidate_calibrator),
            },
        }
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    bundle = {
        "status": "frozen_validated_user_override" if args.user_override else "frozen_validated",
        "winner": args.candidate,
        "calibration": str(candidate_row["calibration"]),
        "feature_count": int(json.loads(FEATURES_PATH.read_text(encoding="utf-8"))["feature_count"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "classifier": CLASSIFIER_PATH.name,
            "calibrator": CALIBRATOR_PATH.name,
            "features": FEATURES_PATH.name,
            "metadata": METADATA_PATH.name,
        },
        "sha256": {
            "classifier": _sha256(CLASSIFIER_PATH),
            "calibrator": _sha256(CALIBRATOR_PATH),
            "features": _sha256(FEATURES_PATH),
            "metadata": _sha256(METADATA_PATH),
            "dataset": _sha256(DATASET_PATH),
        },
        "replacement_decision": {
            "decision": "user_override",
            "reason": "explicit user instruction to replace incumbent",
            "automatic_study_decision": automatic_decision,
        },
    }
    BUNDLE_PATH.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    promotion_record = {
        "promoted_candidate": args.candidate,
        "calibration": str(candidate_row["calibration"]),
        "user_override": bool(args.user_override),
        "candidate_test_metrics": candidate_row,
        "automatic_study_decision": automatic_decision,
        "promoted_at_utc": bundle["created_at_utc"],
        "bundle_sha256": _sha256(BUNDLE_PATH),
    }
    (STUDY_DIR / "promotion.json").write_text(
        json.dumps(promotion_record, indent=2, default=str), encoding="utf-8"
    )
    if REMAINING_METADATA_PATH.exists():
        remaining_metadata = json.loads(REMAINING_METADATA_PATH.read_text(encoding="utf-8"))
        remaining_metadata["exceedance_bundle_sha256"] = _sha256(BUNDLE_PATH)
        remaining_metadata["exceedance_winner"] = args.candidate
        remaining_metadata["exceedance_dependency_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        REMAINING_METADATA_PATH.write_text(
            json.dumps(remaining_metadata, indent=2, default=str), encoding="utf-8"
        )

    predictor = load_hurdle_predictor(
        CLASSIFIER_PATH,
        FEATURES_PATH,
        CALIBRATOR_PATH,
        str(candidate_row["calibration"]),
    )
    sample = pd.read_csv(DATASET_PATH, nrows=5)
    probability = predictor.predict_proba(sample)
    if not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Promoted candidate produced invalid probabilities")
    print(f"Promoted {args.candidate} with {candidate_row['calibration']} calibration")
    print(f"Classifier SHA-256: {bundle['sha256']['classifier']}")
    print(f"Sample probabilities: {probability.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
