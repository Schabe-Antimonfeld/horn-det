"""Matched-window, one-to-one event and all-case localization metrics, chapter 12."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def fraction(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def counts(tp, fp, fn, tn=None):
    result = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": fraction(tp, tp + fp),
              "recall": fraction(tp, tp + fn), "f1": fraction(2 * tp, 2 * tp + fp + fn)}
    if tn is not None:
        result["tn"] = int(tn)
        result["accuracy"] = fraction(tp + tn, tp + tn + fp + fn)
    return result


def window_metrics(labels, probabilities, threshold=0.5):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.shape != probabilities.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Window labels and probabilities must match")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1) or not 0 <= threshold <= 1:
        raise ValueError("Invalid window probabilities or threshold")
    predicted = probabilities >= threshold
    positive = labels == 1
    return counts(np.sum(predicted & positive), np.sum(predicted & ~positive),
                  np.sum(~predicted & positive), np.sum(~predicted & ~positive))


def choose_threshold(labels, probabilities, split="validation"):
    if split != "validation":
        raise ValueError("Choose thresholds on validation only")
    if len(np.unique(labels)) < 2:
        raise ValueError("Threshold selection requires both validation classes")
    best = None
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = window_metrics(labels, probabilities, threshold)
        key = (metrics["f1"] or 0, metrics["recall"] or 0, -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    return {"threshold": best[1], "metrics": best[2], "source": "validation"}


def localization_metrics(truth, estimates, valid):
    truth, estimates, valid = np.asarray(truth), np.asarray(estimates), np.asarray(valid, dtype=bool)
    if truth.shape != estimates.shape or truth.shape != valid.shape or not np.isfinite(truth).all():
        raise ValueError("Localization arrays must match and truth must be finite")
    if not np.isfinite(estimates[valid]).all():
        raise ValueError("Valid angles must be finite")
    errors = np.abs((estimates[valid] - truth[valid] + 180) % 360 - 180)
    return {"total": int(len(truth)), "valid": int(valid.sum()), "invalid": int((~valid).sum()),
            "valid_fraction": fraction(int(valid.sum()), len(truth)),
            "mae_deg": float(errors.mean()) if len(errors) else None,
            "medae_deg": float(np.median(errors)) if len(errors) else None,
            "p90_deg": float(np.percentile(errors, 90)) if len(errors) else None,
            "sr10": fraction(int(np.sum(errors < 10)), len(truth))}


@dataclass(frozen=True)
class Event:
    start: float
    end: float
    confirmed: float | None = None
    event_id: int = 0
    output: float | None = None
    cold: bool = False
    candidate_output: float | None = None
    capture_simulated: float | None = None

    def __post_init__(self):
        if not np.isfinite([self.start, self.end]).all() or not 0 <= self.start <= self.end:
            raise ValueError("Invalid event interval")
        for value in (self.confirmed, self.output, self.candidate_output, self.capture_simulated):
            if value is not None and (not np.isfinite(value) or value < self.start):
                raise ValueError("Confirmation/output cannot precede the candidate")


def event_metrics(labels, predictions, tolerance=0.5, require_overlap=True, duration=None):
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Onset tolerance must be finite and nonnegative")
    predictions = [event for event in predictions if event.confirmed is not None]
    matched = []
    if labels and predictions:
        penalty = len(labels) + len(predictions) + 1
        cost = np.full((len(predictions), len(labels)), float(penalty))
        for row, prediction in enumerate(predictions):
            for column, label in enumerate(labels):
                difference = abs(prediction.start - label.start)
                overlap = min(prediction.end, label.end) > max(prediction.start, label.start)
                if difference <= tolerance and (overlap or not require_overlap):
                    cost[row, column] = difference / (tolerance + 1e-12)
        rows, columns = linear_sum_assignment(cost)
        matched = [(int(row), int(column)) for row, column in zip(rows, columns) if cost[row, column] < penalty]
    tp = len(matched)
    result = counts(tp, len(predictions) - tp, len(labels) - tp)
    result["matches"] = [{"prediction": row, "label": column} for row, column in matched]
    result["false_positive_per_hour"] = None
    result["continuous_duration_s"] = duration
    result["nonhorn_duration_s"] = None
    if duration is not None:
        if not np.isfinite(duration) or duration <= 0 or any(event.end > duration for event in labels):
            raise ValueError("Continuous duration must include all labels and be positive")
        intervals = sorted((event.start, event.end) for event in labels)
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        result["nonhorn_duration_s"] = duration - sum(end - start for start, end in merged)
        result["false_positive_per_hour"] = (len(predictions) - tp) * 3600 / duration
    result["short_horn_recall"] = fraction(sum(labels[column].end - labels[column].start <= 0.3 for _, column in matched),
                                            sum(label.end - label.start <= 0.3 for label in labels))
    result["matched_count"] = tp
    negative = []
    for name, accessor in (("candidate_input", lambda event: event.start),
                           ("candidate_output", lambda event: event.candidate_output),
                           ("confirmed_input", lambda event: event.confirmed),
                           ("confirmed_output", lambda event: event.output),
                           ("capture_simulated", lambda event: event.capture_simulated)):
        values = {"cold": [], "steady": []}
        for row, column in matched:
            endpoint = accessor(predictions[row])
            if endpoint is None:
                continue
            delta = endpoint - labels[column].start
            if delta < 0:
                negative.append({"metric": name, "prediction": row, "label": column, "seconds": delta})
            else:
                values["cold" if predictions[row].cold else "steady"].append(delta)
        result[name + "_latency_s"] = {key: latency_summary(value) for key, value in values.items()}
    result["negative_latency_cases"] = negative
    return result


def latency_summary(values):
    values = np.asarray(values, dtype=float)
    return {"count": int(len(values)), "mean": float(values.mean()) if len(values) else None,
            "p90": float(np.percentile(values, 90)) if len(values) else None,
            "maximum": float(values.max()) if len(values) else None}
