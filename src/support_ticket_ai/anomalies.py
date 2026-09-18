"""Explainable anomaly rules with a stable global statistical baseline."""

import pandas as pd

from support_ticket_ai.analytics import target_rows
from support_ticket_ai.dataset import Dataset
from support_ticket_ai.query import criteria, records, select
from support_ticket_ai.schemas import AnomalySpec, Calculation


def detect(dataset: Dataset, spec: AnomalySpec, as_of: pd.Timestamp) -> Calculation:
    source = dataset.frame(as_of)
    selected = select(source, spec.selection, as_of)
    results = []
    summary: dict = {"long_resolution_count": 0, "overdue_high_priority_count": 0}
    warnings = []
    if spec.rule in {"all", "long_resolution"}:
        baseline = source.loc[source.status.eq("Resolved"), "resolution_time_hrs"].dropna()
        q1, q3 = baseline.quantile([0.25, 0.75]) if len(baseline) else (0.0, 0.0)
        iqr = q3 - q1
        summary["baseline_sample_count"] = len(baseline)
        summary["resolution_threshold_hours"] = None
        if len(baseline) < 4 or iqr == 0:
            warnings.append(
                "Insufficient statistical baseline: need four resolved durations and nonzero IQR."
            )
        else:
            threshold = float(q3 + 1.5 * iqr)
            flagged = selected[
                selected.status.eq("Resolved") & selected.resolution_time_hrs.gt(threshold)
            ].copy()
            flagged["rule"] = "long_resolution"
            flagged["observed_hours"] = flagged.resolution_time_hrs
            flagged["threshold_hours"] = threshold
            flagged["excess_hours"] = flagged.resolution_time_hrs - threshold
            flagged["reason"] = "Resolution duration exceeds the global Q3 + 1.5 × IQR threshold."
            summary["resolution_threshold_hours"] = threshold
            summary["long_resolution_count"] = len(flagged)
            results.extend(records(flagged))
    if spec.rule in {"all", "overdue_high_priority"}:
        checked = target_rows(selected, "resolution", 24)
        # Status conflicts are disclosed separately; recorded unresolved status governs this rule.
        mask = (
            checked.status.isin(["Open", "Escalated"])
            & checked.priority.isin(["High", "Critical"])
            & checked.age_hours.gt(24)
        )
        flagged = checked[mask].copy()
        flagged["rule"] = "overdue_high_priority"
        flagged["observed_hours"] = flagged.age_hours
        flagged["excess_hours"] = flagged.age_hours - 24
        flagged["reason"] = "Recorded unresolved High/Critical ticket is older than 24 hours."
        summary["overdue_high_priority_count"] = len(flagged)
        results.extend(records(flagged))
    results.sort(key=lambda row: (-row["excess_hours"], row["ticket_id"], row["rule"]))
    summary["flag_count"] = len(results)
    summary["unique_ticket_count"] = len({row["ticket_id"] for row in results})
    return Calculation(
        kind="anomalies",
        summary=summary,
        rows=results,
        criteria=criteria(spec.selection, as_of),
        warnings=warnings,
    )
