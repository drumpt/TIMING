from __future__ import annotations

import os
import random
import logging
import pathlib

import numpy as np
import pandas as pd
import torch


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info(f"set seed as {seed}")


def resolve_device(device: torch.device | str | None) -> torch.device:
    """
    Resolve the torch device.
    """
    if device is None:
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        return torch.device(device)


def aggregate_scores(scores: np.ndarray, aggregate_method: str) -> np.ndarray:
    """
    Run aggregation of the WinIT importance scores. If the importance scores is rank 3, return
    the scores.

    Args:
        scores:
            The input importance scores. Shape = (num_samples, num_features, num_times, window_size)
            or (num_samples, num_features, num_times).
        aggregate_method:
            The aggregation method of WinIT

    Returns:
        The aggregate scores as numpy array.
    """
    if scores.ndim == 3:
        return scores

    num_samples, num_features, num_times, window_size = scores.shape
    # where scores[i, :, k] is the importance score window with shape (num_features, window_size)
    # for the prediction at time (k). So scores[i, j, k, l] is the importance of observation
    # (i, j, k - window_size + l + 1) to the prediction at time (k)
    aggregated_scores = np.zeros((num_samples, num_features, num_times))
    for t in range(num_times):
        # windows where obs is included
        relevant_windows = np.arange(t, min(t + window_size, num_times))
        # relative position of obs within window
        relevant_obs = -relevant_windows + t - 1
        relevant_scores = scores[:, :, relevant_windows, relevant_obs]
        relevant_scores = np.nan_to_num(relevant_scores)
        if aggregate_method == "absmax":
            score_max = relevant_scores.max(axis=-1)
            score_min = relevant_scores.min(axis=-1)
            aggregated_scores[:, :, t] = np.where(
                -score_min > score_max, score_min, score_max
            )
        elif aggregate_method == "max":
            aggregated_scores[:, :, t] = relevant_scores.max(axis=-1)
        elif aggregate_method == "mean":
            aggregated_scores[:, :, t] = relevant_scores.mean(axis=-1)
        else:
            raise NotImplementedError(
                f"Aggregation method {aggregate_method} unrecognized"
            )

    return aggregated_scores


def append_df_to_csv(df: pd.DataFrame, csv_path: pathlib.Path) -> int:
    """
    Write intermediate results to CSV. If there is no existing file, create a new one.
    If a file exists, rename the existing file with an incrementing number and write
    the new data to the original path.

    Args:
        df:
            The dataframe we wish to write.
        csv_path:
            The path and the filename of the file.

    Returns:
        An Error code.
        0 - Successful write to original path
        1 - Successful write with existing file renamed
        2 - File exists but read/rename failed
        3 - New file created at original path
    """
    log = logging.getLogger("Utils")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and csv_path.is_file():
        try:
            # Try to read existing file to ensure it's valid
            old_df = pd.read_csv(csv_path)

            # Find available backup name
            i = 1
            while True:
                old_name = csv_path.stem
                backup_path = csv_path.with_name(f"{old_name} ({i}){csv_path.suffix}")
                if not backup_path.exists():
                    break
                i += 1

            # Rename existing file
            csv_path.rename(backup_path)
            log.info(f"Renamed existing file to {backup_path}")

            # Write new data to original path
            df.to_csv(csv_path, index=False)
            return 1

        except Exception as e:
            log.error(f"Error handling existing file: {str(e)}")
            return 2
    else:
        # Create new file at original path
        log.info(f"Creating new file at {csv_path}")
        df.to_csv(csv_path, index=False)
        return 3


def calculate_metrics(y_true, y_pred_proba, threshold):
    """
    Calculate classification metrics for a given threshold.

    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred_proba: Array of predicted probabilities
        threshold: Classification threshold between 0 and 1

    Returns:
        Dictionary with metrics (accuracy, precision, recall, f1)
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate confusion matrix
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / len(y_true)
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def find_optimal_threshold(y_true, y_pred_proba, metric="f1"):
    """
    Find optimal threshold based on specified metric.

    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred_proba: Array of predicted probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')

    Returns:
        optimal_threshold, metrics_at_optimal_threshold
    """
    thresholds = np.linspace(0, 1, 100)
    metrics_list = [calculate_metrics(y_true, y_pred_proba, t) for t in thresholds]

    # Find threshold that maximizes the specified metric
    optimal_idx = np.argmax([m[metric] for m in metrics_list])
    return thresholds[optimal_idx], metrics_list[optimal_idx]
