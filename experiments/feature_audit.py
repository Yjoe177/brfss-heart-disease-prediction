"""Header-level audit of target and cardiovascular history columns.

Run:
    python experiments/feature_audit.py

This uses only the standard library and the official CSV headers. It checks the
documented target name and the three cardiovascular-history columns discussed
in the report. It is not a proof that no derived relationship exists.
"""

import argparse
import csv
import json
import os


def find_data_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("PROJECT1_DATA_DIR"),
        os.path.join(os.getcwd(), "data", "dataset_to_release"),
        os.path.join(here, "..", "data", "dataset_to_release"),
        os.path.join(here, "..", "..", "data", "dataset_to_release"),
    ]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "x_train.csv")):
            return os.path.normpath(path)
    raise FileNotFoundError(
        "Could not find data/dataset_to_release. Set --data-dir or "
        "PROJECT1_DATA_DIR, or place the official data beside run.py.")


def read_header(path):
    with open(path, "r", newline="") as handle:
        return next(csv.reader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results",
        "feature_audit.json"))
    args = parser.parse_args()
    data_dir = args.data_dir or find_data_dir()
    feature_header = read_header(os.path.join(data_dir, "x_train.csv"))
    target_header = read_header(os.path.join(data_dir, "y_train.csv"))
    features = feature_header[1:]
    target = "_MICHD"
    checks = {
        "target_column_present_in_y_train": target in target_header,
        "target_not_among_feature_columns": target not in features,
        "CVDINFR4_absent": "CVDINFR4" not in features,
        "CVDCRHD4_absent": "CVDCRHD4" not in features,
        "CVDSTRK3_present": "CVDSTRK3" in features,
    }
    result = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "feature_columns": len(features),
        "x_train_header": feature_header,
        "y_train_header": target_header,
        "target": target,
        "note": (
            "Header-level audit only. It confirms the documented target and "
            "cardiovascular-history column names; it does not prove the "
            "absence of every possible derived-label relationship. "
            "CVDSTRK3 is a legitimate health-history predictor, not the target."
        ),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
