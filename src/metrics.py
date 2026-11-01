"""Metrics for binary VSR predictions.

Each record has a boolean label, a boolean or None prediction,
and a nonempty relation string. None means the model output could not
be parsed. An invalid prediction is scored as an error and also counted
separately.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def _divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute overall and per-relation metrics from parsed predictions.

    True is the positive class. Invalid predictions (None) count as false
    negatives for true labels and false positives for false labels. This
    keeps the confusion-matrix total equal to the number of input records.
    """
    counts = Counter()
    relation_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for index, record in enumerate(records):
        label = record.get("label")
        prediction = record.get("prediction")
        relation = record.get("relation")
        if type(label) is not bool:
            raise ValueError(f"Record {index}: label must be bool")
        if prediction is not None and type(prediction) is not bool:
            raise ValueError(f"Record {index}: prediction must be bool or None")
        if not isinstance(relation, str) or not relation.strip():
            raise ValueError(f"Record {index}: relation must be a nonempty string")

        relation = relation.strip()
        if prediction is None:
            counts["invalid"] += 1
            outcome = "fn" if label else "fp"
        elif label and prediction:
            outcome = "tp"
        elif not label and not prediction:
            outcome = "tn"
        elif label:
            outcome = "fn"
        else:
            outcome = "fp"
        counts[outcome] += 1
        relation_counts[relation]["total"] += 1
        relation_counts[relation]["correct"] += outcome in ("tp", "tn")

    total = sum(counts[key] for key in ("tp", "tn", "fp", "fn"))
    if total == 0:
        raise ValueError("At least one prediction record is required")

    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    return {
        "sample_count": total,
        "accuracy": _divide(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": _divide(2 * precision * recall, precision + recall),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "invalid_count": counts["invalid"],
        "invalid_rate": _divide(counts["invalid"], total),
        "by_relation": {
            relation: {
                "sample_count": values["total"],
                "correct_count": values["correct"],
                "accuracy": _divide(values["correct"], values["total"]),
            }
            for relation, values in sorted(relation_counts.items())
        },
    }
