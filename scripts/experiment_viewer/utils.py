"""Score extraction and envelope curve for experiment viewer (cloudcast: lower is better)."""
import math
from typing import Dict, List, Any, Union

# Log size limit shared by all interfaces
MAX_LOG_CHARS = 300_000
# Max iterations shown in table before truncation
MAX_ITERATIONS_DISPLAY = 100


def extract_scores_from_iterations(data: Union[Dict, List]) -> List[float]:
    """Extract scores from all_iterations; handle -inf/inf by using a sentinel for envelope."""
    scores: List[float] = []
    if isinstance(data, dict) and "all_iterations" in data:
        for item in data["all_iterations"]:
            if isinstance(item, dict) and "score" in item:
                s = item["score"]
                if s is None:
                    continue
                try:
                    scores.append(float(s))
                except (TypeError, ValueError):
                    continue
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "score" in item:
                s = item["score"]
                if s is None:
                    continue
                try:
                    scores.append(float(s))
                except (TypeError, ValueError):
                    continue
    return scores


def get_envelope_curve(scores: List[float], is_maximize: bool = False) -> List[float]:
    """Best-so-far envelope. For cloudcast (minimize) use is_maximize=False.

    Returns [] if all scores are invalid (±inf or NaN).
    """
    if not scores:
        return []
    inf = float("inf")
    neg_inf = float("-inf")

    def _is_invalid(s: float) -> bool:
        return s == neg_inf or s == inf or (isinstance(s, float) and math.isnan(s))

    if is_maximize:
        out = [scores[0]]
        for s in scores[1:]:
            if _is_invalid(s):
                out.append(out[-1])
            else:
                out.append(max(out[-1], s))
        return out
    else:
        # Minimize: skip ±inf/NaN (treat as failed, keep previous best)
        out = [scores[0] if not _is_invalid(scores[0]) else inf]
        for s in scores[1:]:
            if _is_invalid(s):
                out.append(out[-1])
            else:
                out.append(min(out[-1], s))
        # Propagate first valid score backwards; return [] if all invalid
        first_valid = next((i for i, v in enumerate(out) if not _is_invalid(v)), None)
        if first_valid is None:
            return []
        for j in range(first_valid):
            out[j] = out[first_valid]
        return out


def extract_baselines_dict(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract baseline name -> score from data['baselines']."""
    result: Dict[str, float] = {}
    if not isinstance(data, dict) or "baselines" not in data:
        return result
    for key, val in data["baselines"].items():
        if isinstance(val, dict) and "score" in val:
            try:
                s = float(val["score"])
                if math.isfinite(s):
                    result[key] = s
            except (TypeError, ValueError):
                pass
    return result
