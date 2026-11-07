"""Train the final BRFSS model and write its submission CSV.

All preprocessing is fitted on labeled training rows. The threshold was fixed
on a separate training-side calibration split before the final all-data fit.
"""

import argparse
import gc
import os
import time

import numpy as np

import pipeline


DECISION_THRESHOLD = 0.21475973018819802


def write_submission(path, ids, predictions):
    """Write predictions in the course's required order and format."""
    with open(path, "w", newline="") as handle:
        handle.write("Id,Prediction\n")
        for row_id, prediction in zip(ids, predictions):
            handle.write("%d,%d\n" % (int(row_id), int(prediction)))


def verify_submission(path, test_ids):
    """Check the saved file against the original test IDs."""
    with open(path, "r") as handle:
        header = handle.readline().strip()
        rows = [line.strip().split(",") for line in handle if line.strip()]

    ids = np.array([int(row[0]) for row in rows], dtype=np.int64)
    predictions = np.array([int(row[1]) for row in rows], dtype=np.int64)
    if header != "Id,Prediction":
        raise RuntimeError("Incorrect submission header.")
    if ids.size != test_ids.size:
        raise RuntimeError("Incorrect number of predictions.")
    if not np.array_equal(ids, test_ids):
        raise RuntimeError("Test IDs are missing, duplicated or reordered.")
    if not np.all(np.isin(predictions, (-1, 1))):
        raise RuntimeError("Predictions must be -1 or +1.")
    print("Submission verification passed: %d rows, %d positive." % (
        ids.size, int((predictions == 1).sum())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=None,
        help="directory containing x_train.csv, y_train.csv, and x_test.csv",
    )
    parser.add_argument(
        "--output", default=None,
        help="submission CSV path (default: results/submission.csv)",
    )
    args = parser.parse_args()

    start = time.time()
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or pipeline.find_data_dir()
    output_path = args.output or os.path.join(root, "results", "submission.csv")

    x_train, y_train, header = pipeline.load_train(data_dir)
    print("Loaded training %s." % (x_train.shape,))

    rows = np.arange(x_train.shape[0], dtype=np.int64)
    candidate = pipeline.candidate_columns(header)
    preprocessing = pipeline.fit_preprocessor(x_train, rows, candidate)
    design_train = pipeline.make_design(x_train, rows, preprocessing)
    del x_train, rows
    gc.collect()
    summary = pipeline.encoded_summary(preprocessing, design_train.shape[1])
    print("Design dimension: %(design_dimension)d; %(n_input_features)d input features." % summary)

    labels = (y_train == 1).astype(float)
    fitted = pipeline.train_sgd(
        design_train, labels, lr0=0.01, epochs=90, schedule="inverse",
        lam=1e-5, seed=2027, decay=0.1, progress=True)
    del design_train, labels
    gc.collect()

    x_test, test_ids, test_header = pipeline.load_test(data_dir)
    if header[1:] != test_header[1:]:
        raise RuntimeError("Training and test feature columns do not match.")
    design_test = pipeline.make_design(
        x_test, np.arange(x_test.shape[0], dtype=np.int64), preprocessing)
    del x_test
    print("Loaded test %s." % (design_test.shape,))
    probabilities = pipeline.sigmoid(design_test @ fitted["w"])
    predictions = np.where(probabilities >= DECISION_THRESHOLD, 1, -1).astype(int)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    write_submission(output_path, test_ids, predictions)
    verify_submission(output_path, test_ids)
    print("Wrote %s; final loss %.6f; positives %d (%.3f%%)." % (
        output_path, fitted["data_loss"], int((predictions == 1).sum()),
        100.0 * float((predictions == 1).mean())))
    print("Total runtime: %.1fs." % (time.time() - start))


if __name__ == "__main__":
    main()
