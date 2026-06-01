#!/usr/bin/env python3
"""Gradio dashboard for exploring and comparing GRPO training runs.

Launch:
    python tools/grpo_dashboard.py [--logs-dir /mnt/data/pmartins/logs_grpo] [--port 7860]
"""

import argparse
import glob
import json
import os
import re
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import gradio as gr
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LOGS_DIR = "/mnt/data/pmartins/logs_grpo"

# Known task types and their plottable numeric score fields (short names).
# Unknown task types are discovered dynamically from reward_details keys.
KNOWN_TASK_COMPONENTS = {
    "repetition": ["detection", "correctness", "severity", "format"],
    "full_exercise": ["correctness", "severity", "format"],
    "comparison": ["correctness", "format"],
    "video_mcqa": ["correctness", "format"],
    "image_mcqa": ["correctness", "format"],
    "phase_sequencing_mcqa": ["correctness", "format"],
    "muscle_exercise_mcqa": ["correctness", "format"],
    "error_correction_mcqa": ["correctness", "format"],
    "error_recognition": ["correctness", "format"],
    "exercise_name_identification": ["fuzzy_score", "exact_match"],
    "keypoint_prediction": ["mean_oks", "detection_rate"],
    "keypoint_labeling": ["f1", "precision", "recall", "exact_match"],
    # Generic MCQ (when specific MCQ subtype can't be determined from old logs)
    "mcq": ["correctness", "format"],
}

# Non-score detail fields shown in rollout browser (not aggregated/plotted)
EXTRA_DETAIL_FIELDS = {
    "repetition": ["gt_effectiveness", "gt_injury_risk", "pred_effectiveness", "pred_injury_risk"],
    "full_exercise": ["gt_effectiveness", "gt_injury_risk", "pred_effectiveness", "pred_injury_risk"],
    "comparison": ["gt_verdict", "pred_verdict"],
    "video_mcqa": ["gt_answer", "pred_answer"],
    "image_mcqa": ["gt_answer", "pred_answer"],
    "phase_sequencing_mcqa": ["gt_answer", "pred_answer"],
    "muscle_exercise_mcqa": ["gt_answer", "pred_answer"],
    "error_correction_mcqa": ["gt_answer", "pred_answer"],
    "error_recognition": ["gt_answer", "pred_answer"],
    "exercise_name_identification": ["gt_name", "pred_name"],
    "keypoint_prediction": ["gt_keypoints", "pred_keypoints", "matched"],
    "keypoint_labeling": ["gt_labels", "pred_labels"],
    "mcq": ["gt_answer", "pred_answer"],
}

# Old-format field name mapping (legacy logs stored "detection_score" at top level)
OLD_FIELD_MAP = {
    "detection_score": "detection",
    "correctness_score": "correctness",
    "severity_score": "severity",
    "format_score": "format",
}

# Distinct colors for overlay plots
COLORS = [
    "#1976d2", "#d32f2f", "#388e3c", "#f57c00", "#7b1fa2",
    "#00838f", "#c2185b", "#455a64", "#6d4c41", "#afb42b",
]

TASK_TYPE_LABELS = {
    "repetition": "Repetition Analysis",
    "full_exercise": "Full Exercise Analysis",
    "comparison": "Comparison",
    "video_mcqa": "Video MCQ",
    "image_mcqa": "Image MCQ",
    "phase_sequencing_mcqa": "Phase Sequencing MCQ",
    "muscle_exercise_mcqa": "Muscle Exercise MCQ",
    "error_correction_mcqa": "Error Correction MCQ",
    "error_recognition": "Error Recognition",
    "exercise_name_identification": "Exercise Name ID",
    "keypoint_prediction": "Keypoint Prediction",
    "keypoint_labeling": "Keypoint Labeling",
    "mcq": "MCQ",
}

# Fields that should NOT be treated as score fields when auto-discovering from reward_details
_NON_SCORE_KEYS = {
    "task_type", "final_reward", "gt_effectiveness", "gt_injury_risk",
    "pred_effectiveness", "pred_injury_risk", "gt_errors", "pred_errors",
    "gt_answers", "pred_answers", "gt_answer", "pred_answer",
    "gt_verdict", "pred_verdict", "gt_name", "pred_name",
    "gt_keypoints", "pred_keypoints", "matched", "gt_labels", "pred_labels",
    "error",
}


def get_task_components(task_type):
    """Return score component names for a task type from the registry."""
    return KNOWN_TASK_COMPONENTS.get(task_type, [])


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def shorten_run_name(name):
    """Shorten a run name by stripping common boilerplate."""
    s = name
    s = re.sub(r'^grpo_sft_', '', s)
    # Remove dataset descriptors entirely (the date-tagged data portions)
    s = re.sub(r'_?(?:merged_rep|repetition_feedback_severity|repetition)_\d{4}(?:_10k)?(?:_full_exercise_feedback_\d{4})?(?:_reasoning)?', '', s)
    # Remove duplicate remaining patterns
    s = re.sub(r'(_[^_]+)\1+', r'\1', s)
    # Abbreviate reward mode keywords
    s = s.replace('detection_correctness_severity', 'dcs')
    s = s.replace('detection_severity', 'ds')
    s = s.replace('_multiplicative', '_mult')
    s = s.replace('_format_', '_f')
    # Re-add key dataset tags from original name
    tags = []
    if 'merged' in name or 'mixed' in name:
        tags.append('mixed')
    else:
        tags.append('rep')
    if 'reasoning' in name:
        tags.append('reas')
    if '10k' in name:
        tags.append('10k')
    if 'comparison' in name:
        tags.append('comp')
    if 'g16' in name:
        tags.append('g16')
    # Extract model size
    model = re.search(r'qwen[_]?3[5s]?[_]?(?:vl[_]?)?(\d+b)', name)
    model_str = model.group(1) if model else '?b'
    # Clean up remaining reward config
    s = re.sub(r'^qwen[_]?3[5s]?[_]?(?:vl[_]?)?\d+b[_]?', '', s)
    # Remove redundant tags already captured above
    s = re.sub(r'_?comparison_reps_\d+', '', s)
    s = re.sub(r'_?g16', '', s)
    s = re.sub(r'__+', '_', s).strip('_')
    return f"{model_str} {','.join(tags)} - {s}" if s else f"{model_str} {','.join(tags)}"


def _build_short_name_map(full_names):
    """Build a mapping from full run name to unique short name."""
    short_to_fulls = {}
    for name in full_names:
        short = shorten_run_name(name)
        short_to_fulls.setdefault(short, []).append(name)

    result = {}
    for short, fulls in short_to_fulls.items():
        if len(fulls) == 1:
            result[fulls[0]] = short
        else:
            for i, full in enumerate(sorted(fulls)):
                result[full] = f"{short} #{i+1}"
    return result


def discover_runs(logs_dir):
    """Return sorted list of (run_name, [exp_paths]) for all valid runs.

    All exp_NNN folders containing train_data_step*.jsonl are returned for
    each run, sorted ascending by exp id so resumed-run data (higher exp id)
    overrides earlier exp folders on step-number conflicts.
    """
    runs = []
    for name in sorted(os.listdir(logs_dir)):
        run_dir = os.path.join(logs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        exp_dirs = sorted(glob.glob(os.path.join(run_dir, "exp_*")))
        exp_dirs_with_data = [
            d for d in exp_dirs
            if glob.glob(os.path.join(d, "train_data_step*.jsonl"))
        ]
        if exp_dirs_with_data:
            runs.append((name, exp_dirs_with_data))
    return runs


def _infer_task_type_from_details(details):
    """Infer the task type from the keys present in a reward_details dict."""
    keys = set(details.keys())
    if "task_type" in details and details["task_type"]:
        return details["task_type"]
    # Repetition: has detection + gt_errors
    if "detection" in keys and "gt_errors" in keys:
        return "repetition"
    # Full exercise: has gt_answers
    if "gt_answers" in keys:
        return "full_exercise"
    # Comparison: has gt_verdict
    if "gt_verdict" in keys:
        return "comparison"
    # Keypoint prediction: has mean_oks
    if "mean_oks" in keys:
        return "keypoint_prediction"
    # Keypoint labeling: has f1 + precision + recall
    if "f1" in keys and "precision" in keys:
        return "keypoint_labeling"
    # Exercise name: has fuzzy_score
    if "fuzzy_score" in keys:
        return "exercise_name_identification"
    # MCQ: has gt_answer + pred_answer
    if "gt_answer" in keys and "pred_answer" in keys:
        return "mcq"
    return "unknown"


def _extract_record_scores(rec):
    """Extract task_type and score values from a single JSONL record.

    Handles both new format (scores in reward_details) and old format
    (scores as top-level fields with _score suffix).

    Returns (task_type, score_values_dict, score_fields_list).
    """
    # Try new format first: scores nested in reward_details
    rd = rec.get("reward_details")
    if rd and isinstance(rd, list) and rd[0] and isinstance(rd[0], dict):
        details = rd[0]
        # Task type from reward_details, then top-level, then infer from keys
        task_type = details.get("task_type")
        if not task_type:
            tt_field = rec.get("task_type")
            task_type = (tt_field[0] if isinstance(tt_field, list) else tt_field) if tt_field else None
        if not task_type:
            task_type = _infer_task_type_from_details(details)

        score_fields = get_task_components(task_type)
        if not score_fields:
            # Auto-discover numeric fields for unknown task types
            score_fields = [
                k for k, v in details.items()
                if isinstance(v, (int, float)) and k not in _NON_SCORE_KEYS
            ]

        score_values = {}
        for f in score_fields:
            val = details.get(f)
            score_values[f] = float(val) if val is not None else float("nan")

        return task_type, score_values, score_fields

    # Old format: scores at top level with "_score" suffix
    tt_field = rec.get("task_type")
    task_type = (tt_field[0] if isinstance(tt_field, list) else tt_field) if tt_field else None
    if task_type is None:
        det = rec.get("detection_score", [None])[0]
        corr = rec.get("correctness_score", [None])[0]
        if det is not None:
            task_type = "repetition"
        elif corr is not None:
            task_type = "full_exercise"
        else:
            task_type = "unknown"

    score_fields = get_task_components(task_type)
    if not score_fields:
        # For old format unknown types, try the legacy 4-field set
        score_fields = list(OLD_FIELD_MAP.values())

    score_values = {}
    for new_name in score_fields:
        old_name = new_name + "_score"
        if old_name in rec:
            val = rec[old_name]
            val = val[0] if isinstance(val, list) else val
            score_values[new_name] = float(val) if val is not None else float("nan")
        elif new_name in rec:
            val = rec[new_name]
            val = val[0] if isinstance(val, list) else val
            score_values[new_name] = float(val) if val is not None else float("nan")
        else:
            score_values[new_name] = float("nan")

    return task_type, score_values, score_fields


def _parse_step_file(filepath):
    """Parse a single JSONL step file into per-record metrics, grouped by task_type."""
    records_by_type = {}  # task_type -> {rewards: [], scores: {field: []}, ...}

    with open(filepath) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip corrupted/truncated lines
            reward = rec["rewards"][0]
            sample_id = rec.get("sample_id", [None])[0]

            task_type, score_values, score_fields = _extract_record_scores(rec)

            if task_type not in records_by_type:
                records_by_type[task_type] = {
                    "rewards": [],
                    "sample_ids": [],
                    "scores": {f: [] for f in score_fields},
                    "_score_fields": list(score_fields),
                }

            bucket = records_by_type[task_type]
            bucket["rewards"].append(reward)
            bucket["sample_ids"].append(sample_id)

            # Extend score dict if new fields appear mid-file
            for f in score_fields:
                if f not in bucket["scores"]:
                    bucket["scores"][f] = [float("nan")] * (len(bucket["rewards"]) - 1)
                    bucket["_score_fields"].append(f)
            for f in bucket["scores"]:
                bucket["scores"][f].append(score_values.get(f, float("nan")))

    # Convert to numpy
    result = {}
    for tt, bucket in records_by_type.items():
        entry = {
            "rewards": np.array(bucket["rewards"], dtype=np.float64),
            "sample_ids": bucket["sample_ids"],
            "_score_fields": bucket["_score_fields"],
        }
        for f in bucket["scores"]:
            entry[f] = np.array(bucket["scores"][f], dtype=np.float64)
        result[tt] = entry

    return result


def _parse_weight_digits(digits, n_components):
    """Parse concatenated weight digits into n_components floats.

    Each component is encoded as either 2 digits (0.X) or 3 digits (0.XX).
    E.g. '025025025025' -> [0.25, 0.25, 0.25, 0.25]
         '03020401' -> [0.3, 0.2, 0.4, 0.1]
         '02505025' -> [0.25, 0.5, 0.25]  (mixed 3/2/3)

    Uses recursive search with best-sum-to-1 heuristic.
    """
    def _search(remaining, n_left):
        if n_left == 0:
            return [[]] if remaining == "" else []
        results = []
        for width in (2, 3):
            if len(remaining) >= width:
                chunk = remaining[:width]
                val = int(chunk) / (10 if width == 2 else 100)
                if 0 <= val <= 1:
                    for rest in _search(remaining[width:], n_left - 1):
                        results.append([val] + rest)
        return results

    candidates = _search(digits, n_components)
    if not candidates:
        return None
    # Pick the one whose sum is closest to 1.0
    best = min(candidates, key=lambda c: abs(sum(c) - 1.0))
    if abs(sum(best) - 1.0) > 0.15:
        return None  # too far off, probably a bad parse
    return best


def parse_weights_from_run_name(run_name):
    """Extract reward component weights from run directory name.

    Returns dict with 'repetition', 'full_exercise', 'comparison' keys,
    each containing a dict of component weights.
    """
    weights = {}

    # Repetition weights: 4 components (detection, correctness, severity, format)
    # Patterns: "detection_correctness_severity_XX_multiplicative_format_DIGITS"
    #       or: "dcs_XX_multiplicative_format_DIGITS"
    #       or: "dcs_XX_multiplicative_DIGITS" (no format keyword, format_weight=0)
    #       or: "dcs_XX_format_DIGITS" (non-multiplicative)
    rep_match = re.search(
        r'(?:detection_correctness_severity|dcs)_(\d+)_'
        r'(?:multiplicative_)?'
        r'(?:format_)?'
        r'(\d{6,12})',
        run_name
    )
    if rep_match:
        _error_weights = rep_match.group(1)  # e.g. "21" -> error_w=2, non_error_w=1
        digits = rep_match.group(2)
        parsed = _parse_weight_digits(digits, 4)
        if parsed is None:
            # Try 3 components (no format)
            parsed = _parse_weight_digits(digits, 3)
            if parsed:
                parsed.append(0.0)
        if parsed:
            weights["repetition"] = {
                "detection": parsed[0],
                "correctness": parsed[1],
                "severity": parsed[2],
                "format": parsed[3],
            }

    # Full exercise weights: 3 components (correctness, severity, format)
    # Pattern: "detection_severity_XX_multiplicative_format_DIGITS" or "ds_XX_..."
    fe_match = re.search(
        r'(?:detection_severity|ds)_(\d+)_'
        r'(?:multiplicative_)?'
        r'(?:format_)?'
        r'(\d{4,9})',
        run_name
    )
    if fe_match:
        digits = fe_match.group(2)
        parsed = _parse_weight_digits(digits, 3)
        if parsed:
            weights["full_exercise"] = {
                "correctness": parsed[0],
                "severity": parsed[1],
                "format": parsed[2],
            }

    # Comparison weights: pattern "c_DIGITS" (verdict, format)
    comp_match = re.search(r'_c_(\d{4})', run_name)
    if comp_match:
        digits = comp_match.group(1)
        parsed = _parse_weight_digits(digits, 2)
        if parsed:
            weights["comparison"] = {
                "correctness": parsed[0],
                "format": parsed[1],
            }

    return weights


def backcompute_severity(step_data, weights_by_type):
    """Back-compute severity from reward and other components when it's missing.

    For repetition:
      reward = (dw*det + cw*corr + sw*sev + fw*fmt) / total_w
      => sev = (reward * total_w - dw*det - cw*corr - fw*fmt) / sw

    For full_exercise:
      reward = (cw*corr + sw*sev + fw*fmt) / total_w
      => sev = (reward * total_w - cw*corr - fw*fmt) / sw
    """
    for step in step_data:
        for tt in step_data[step]:
            entry = step_data[step][tt]
            if "severity" not in entry:
                continue
            sev = entry["severity"]

            # Only back-compute if severity is all NaN
            if not np.all(np.isnan(sev)):
                continue

            w = weights_by_type.get(tt)
            if not w or w.get("severity", 0) == 0:
                continue

            sw = w["severity"]
            rewards = entry["rewards"]

            if tt == "repetition":
                dw = w.get("detection", 0)
                cw = w.get("correctness", 0)
                fw = w.get("format", 0)
                total_w = dw + cw + sw + fw

                det = np.where(np.isnan(entry.get("detection", np.array([]))), 0.0, entry.get("detection", np.array([])))
                corr = np.where(np.isnan(entry.get("correctness", np.array([]))), 0.0, entry.get("correctness", np.array([])))
                fmt = np.where(np.isnan(entry.get("format", np.array([]))), 0.0, entry.get("format", np.array([])))

                computed = (rewards * total_w - dw * det - cw * corr - fw * fmt) / sw

            elif tt == "full_exercise":
                cw = w.get("correctness", 0)
                fw = w.get("format", 0)
                total_w = cw + sw + fw

                corr = np.where(np.isnan(entry.get("correctness", np.array([]))), 0.0, entry.get("correctness", np.array([])))
                fmt = np.where(np.isnan(entry.get("format", np.array([]))), 0.0, entry.get("format", np.array([])))

                computed = (rewards * total_w - cw * corr - fw * fmt) / sw
            else:
                continue

            entry["severity"] = np.clip(computed, 0.0, 1.0)

    return step_data


def load_run_metrics(exp_paths, num_workers=8, run_name=None):
    """Load all step files for a run. Returns dict[step_num -> {task_type -> data}].

    Accepts either a single exp path (str) or a list of exp paths. When a
    list is given, later paths overwrite earlier ones for any duplicate step
    number — this lets resumed runs override the original run on overlap.

    If run_name is provided, attempts to parse reward weights from the name
    and back-compute severity where it's missing.
    """
    if isinstance(exp_paths, str):
        exp_paths = [exp_paths]

    step_files = {}
    for exp_path in exp_paths:  # ascending order: later wins on overlap
        for fp in glob.glob(os.path.join(exp_path, "train_data_step*.jsonl")):
            m = re.search(r"step(\d+)", os.path.basename(fp))
            if m:
                step_files[int(m.group(1))] = fp

    steps_sorted = sorted(step_files.keys())
    filepaths = [step_files[s] for s in steps_sorted]

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        results = list(ex.map(_parse_step_file, filepaths))

    step_data = dict(zip(steps_sorted, results))

    # Back-compute severity from reward + other components if missing
    if run_name:
        weights = parse_weights_from_run_name(run_name)
        if weights:
            step_data = backcompute_severity(step_data, weights)

    return step_data


# === SDAR observability: load per-run aggregate metrics emitted by
# sdar.async_sdar_train into <log_dir>/train_metrics.jsonl. One row per
# training step. Returns a list-of-dicts sorted by step. Missing files are
# silently treated as no SDAR data for that run (GRPO-only runs).
def load_run_train_metrics(exp_paths):
    """Load train_metrics.jsonl rows across one or more exp_* folders.

    Each row is a dict with keys like {step, loss, gen_kl_error,
    opsd/loss, opsd/gate_mean, opsd/gate_active_frac,
    opsd/teacher_minus_student_mean, …}. Later exp folders win on
    duplicate step numbers (resumed-run semantics).
    """
    if isinstance(exp_paths, str):
        exp_paths = [exp_paths]
    rows_by_step = {}
    for exp_path in exp_paths:
        fp = os.path.join(exp_path, "train_metrics.jsonl")
        if not os.path.exists(fp):
            continue
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step = int(row.get("step", -1))
                if step < 0:
                    continue
                rows_by_step[step] = row  # later exp overwrites earlier
    return [rows_by_step[s] for s in sorted(rows_by_step)]


def load_step_rollouts(exp_paths, step):
    """Load full rollout records for a single step (for the rollout browser).

    Accepts either a single exp path (str) or a list of exp paths. When a
    list is given, exp folders are searched newest-first so a resumed-run
    file overrides an earlier file with the same step number.
    """
    if isinstance(exp_paths, str):
        exp_paths = [exp_paths]

    for exp_path in reversed(exp_paths):  # newest first
        fp = os.path.join(exp_path, f"train_data_step{step}.jsonl")
        if not os.path.exists(fp):
            continue
        records = []
        with open(fp) as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records
    return []


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def _discover_score_fields(step_data):
    """Collect score field names per task type from parsed step data."""
    fields_by_type = {}
    for step_dict in step_data.values():
        for tt, entry in step_dict.items():
            sf = entry.get("_score_fields", [])
            if tt not in fields_by_type:
                fields_by_type[tt] = list(sf)
            else:
                for f in sf:
                    if f not in fields_by_type[tt]:
                        fields_by_type[tt].append(f)
    return fields_by_type


def aggregate_run(step_data, num_rollouts=8):
    """Compute per-step aggregate metrics, split by task type.

    Returns dict with:
      - "steps": np array of step numbers
      - "all": {mean_reward, mean_std, frac_zero_var, ...} over all records
      - "by_type": {task_type: {mean_reward, mean_<score>, ...}} per task type
      - "task_types": list of task types found
      - "components_by_type": {task_type: [field_names]} discovered from data
    """
    steps = sorted(step_data.keys())

    # Discover all task types and their score fields across steps
    all_task_types = set()
    for s in steps:
        all_task_types.update(step_data[s].keys())
    all_task_types = sorted(all_task_types)

    components_by_type = _discover_score_fields(step_data)

    # Union of all score fields (for combined metrics)
    all_score_fields = []
    for fields in components_by_type.values():
        for f in fields:
            if f not in all_score_fields:
                all_score_fields.append(f)

    # Aggregates over ALL records (combined)
    agg_all = {"mean_reward": [], "mean_std": [], "frac_zero_var": []}
    for f in all_score_fields:
        agg_all[f"mean_{f}"] = []

    # Aggregates per task type
    agg_by_type = {}
    for tt in all_task_types:
        agg_by_type[tt] = {"mean_reward": [], "count": []}
        components = components_by_type.get(tt, [])
        for f in components:
            agg_by_type[tt][f"mean_{f}"] = []
            agg_by_type[tt][f"std_{f}"] = []

    for s in steps:
        type_data = step_data[s]

        # Combined metrics
        all_rewards = np.concatenate([type_data[tt]["rewards"] for tt in type_data])
        n = len(all_rewards)
        n_prompts = n // num_rollouts
        if n_prompts == 0:
            continue

        grouped = all_rewards[:n_prompts * num_rollouts].reshape(n_prompts, num_rollouts)
        prompt_stds = np.std(grouped, axis=1, ddof=1)
        agg_all["mean_reward"].append(np.mean(all_rewards))
        agg_all["mean_std"].append(np.mean(prompt_stds))
        agg_all["frac_zero_var"].append(np.mean(prompt_stds < 1e-8))

        # Combined score means
        for f in all_score_fields:
            vals_list = [type_data[tt][f] for tt in type_data if f in type_data[tt]]
            if vals_list:
                all_vals = np.concatenate(vals_list)
                agg_all[f"mean_{f}"].append(np.nanmean(all_vals) if len(all_vals) > 0 else float("nan"))
            else:
                agg_all[f"mean_{f}"].append(float("nan"))

        # Per task-type metrics
        for tt in all_task_types:
            components = components_by_type.get(tt, [])
            if tt not in type_data:
                agg_by_type[tt]["mean_reward"].append(float("nan"))
                agg_by_type[tt]["count"].append(0)
                for f in components:
                    agg_by_type[tt][f"mean_{f}"].append(float("nan"))
                    agg_by_type[tt][f"std_{f}"].append(float("nan"))
                continue

            rewards_tt = type_data[tt]["rewards"]
            agg_by_type[tt]["mean_reward"].append(np.mean(rewards_tt))
            agg_by_type[tt]["count"].append(len(rewards_tt))

            n_tt = len(rewards_tt)
            n_prompts_tt = n_tt // num_rollouts

            for f in components:
                if f not in type_data[tt]:
                    agg_by_type[tt][f"mean_{f}"].append(float("nan"))
                    agg_by_type[tt][f"std_{f}"].append(float("nan"))
                    continue
                vals = type_data[tt][f]
                agg_by_type[tt][f"mean_{f}"].append(np.nanmean(vals) if len(vals) > 0 else float("nan"))
                # Per-prompt std for this component
                if n_prompts_tt > 0 and len(vals) >= n_prompts_tt * num_rollouts:
                    grouped = vals[:n_prompts_tt * num_rollouts].reshape(n_prompts_tt, num_rollouts)
                    agg_by_type[tt][f"std_{f}"].append(np.nanmean(np.nanstd(grouped, axis=1, ddof=1)))
                else:
                    agg_by_type[tt][f"std_{f}"].append(float("nan"))

    # Convert to numpy
    result = {"steps": np.array(steps), "task_types": all_task_types, "components_by_type": components_by_type}

    for k in agg_all:
        agg_all[k] = np.array(agg_all[k])
    result["all"] = agg_all

    for tt in all_task_types:
        for k in agg_by_type[tt]:
            agg_by_type[tt][k] = np.array(agg_by_type[tt][k])
    result["by_type"] = agg_by_type

    return result


def smooth(arr, w):
    if w <= 1 or len(arr) < w:
        return arr
    # NaN-aware moving average (np.convolve propagates NaN)
    valid = ~np.isnan(arr)
    arr_clean = np.where(valid, arr, 0.0)
    cumsum = np.cumsum(arr_clean)
    cumcount = np.cumsum(valid.astype(np.float64))
    sums = cumsum[w - 1:] - np.concatenate([[0.0], cumsum[:-w]])
    counts = cumcount[w - 1:] - np.concatenate([[0.0], cumcount[:-w]])
    return np.where(counts > 0, sums / counts, np.nan)


def smooth_x(x, w):
    if w <= 1 or len(x) < w:
        return x
    s = smooth(np.zeros(len(x)), w)
    return x[w // 2: w // 2 + len(s)]


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def make_legend_figure(run_aggs):
    """Create a standalone figure containing only the color legend for runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    handles = []
    for i, (label, _) in enumerate(run_aggs):
        handles.append(Line2D([0], [0], color=COLORS[i % len(COLORS)], linewidth=3, label=label))

    n = len(run_aggs)
    n_cols = 1  # one column for readability with long-ish labels
    n_rows = n
    fig_height = max(0.35 * n_rows + 0.5, 1.0)

    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=n_cols,
              fontsize=9, frameon=True, fancybox=True, shadow=False,
              handlelength=3, columnspacing=2)
    fig.tight_layout()
    return fig


def _auto_ylim(ax, margin=0.05):
    """Set y-axis limits based on plotted data with a margin."""
    lines = ax.get_lines()
    if not lines:
        return
    all_y = []
    for line in lines:
        ydata = line.get_ydata()
        if len(ydata) > 0:
            finite = ydata[np.isfinite(ydata)]
            if len(finite) > 0:
                all_y.append(finite)
    if not all_y:
        return
    all_y = np.concatenate(all_y)
    ymin, ymax = np.min(all_y), np.max(all_y)
    span = ymax - ymin if ymax > ymin else 0.1
    ax.set_ylim(ymin - margin * span, ymax + margin * span)


def _make_overlay_plot(run_aggs, key_fn, title, ylabel, smoothing, ylim=None):
    """Generic overlay plot without legend. key_fn(agg) -> array or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (label, agg) in enumerate(run_aggs):
        arr = key_fn(agg)
        if arr is None or len(arr) == 0:
            continue
        sx = smooth_x(agg["steps"], smoothing)
        sy = smooth(arr, smoothing)
        ax.plot(sx, sy, color=COLORS[i % len(COLORS)], linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        _auto_ylim(ax)
    fig.tight_layout()
    return fig


def plot_mean_reward(run_aggs, smoothing):
    return _make_overlay_plot(
        run_aggs, lambda a: a["all"]["mean_reward"],
        "Mean Reward Over Training", "Mean Reward", smoothing)


# === SDAR observability: per-step OPSD metric plots, sourced from each run's
# train_metrics.jsonl (loaded via AppState.get_train_metrics). The helper
# matches the shape of _make_overlay_plot but takes raw row lists instead of
# the task-typed aggregate dict so it stays usable for SDAR-only runs.
def _plot_train_metrics_overlay(
    run_metrics_list, key, title, ylabel, smoothing, ylim=None
):
    """Overlay a single key from each run's train_metrics.jsonl rows.

    Args:
        run_metrics_list: list of (label, list[dict]) — one tuple per run.
        key: the row column to plot (e.g. "opsd/gate_mean").
    Runs without the key produce no line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (label, rows) in enumerate(run_metrics_list):
        if not rows:
            continue
        xs = [r["step"] for r in rows if key in r and r[key] is not None]
        ys = [r[key] for r in rows if key in r and r[key] is not None]
        if not ys:
            continue
        try:
            ys_arr = np.array(ys, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        xs_arr = np.array(xs)
        sx = smooth_x(xs_arr, smoothing)
        sy = smooth(ys_arr, smoothing)
        ax.plot(sx, sy, color=COLORS[i % len(COLORS)], linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        _auto_ylim(ax)
    fig.tight_layout()
    return fig


def plot_opsd_loss(run_metrics_list, smoothing):
    return _plot_train_metrics_overlay(
        run_metrics_list, "opsd/loss",
        "OPSD Loss (unscaled)", "opsd/loss", smoothing)


def plot_opsd_gate_mean(run_metrics_list, smoothing):
    return _plot_train_metrics_overlay(
        run_metrics_list, "opsd/gate_mean",
        "OPSD Gate Mean (per-step average over response tokens)",
        "g_t mean", smoothing, ylim=(0, 1.05))


def plot_opsd_gate_active_frac(run_metrics_list, smoothing):
    return _plot_train_metrics_overlay(
        run_metrics_list, "opsd/gate_active_frac",
        "OPSD Gate Active Fraction (g_t > 0.5)",
        "Fraction of response tokens", smoothing, ylim=(0, 1.05))


def plot_opsd_teacher_minus_student(run_metrics_list, smoothing):
    return _plot_train_metrics_overlay(
        run_metrics_list, "opsd/teacher_minus_student_mean",
        "Teacher − Student Log-Prob Gap (Δ̄)",
        "Δ̄ (teacher − student)", smoothing)
# === END SDAR observability ===


def plot_mean_std(run_aggs, smoothing):
    return _make_overlay_plot(
        run_aggs, lambda a: a["all"]["mean_std"],
        "Reward Diversity (Per-Prompt Std)", "Mean Per-Prompt Std", smoothing)


def plot_zero_var(run_aggs, smoothing):
    return _make_overlay_plot(
        run_aggs, lambda a: a["all"]["frac_zero_var"],
        "Zero-Variance Fraction", "Fraction of Prompts", smoothing, ylim=(0, 1.05))


def plot_mean_reward_by_type(run_aggs, smoothing):
    """One subplot per task type found across runs, showing mean reward."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_types = set()
    for _, agg in run_aggs:
        all_types.update(agg["task_types"])
    all_types = sorted(all_types)

    if not all_types:
        return None

    n_cols = len(all_types)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for j, tt in enumerate(all_types):
        ax = axes[j]
        for i, (label, agg) in enumerate(run_aggs):
            if tt not in agg["by_type"]:
                continue
            arr = agg["by_type"][tt]["mean_reward"]
            if len(arr) == 0:
                continue
            sx = smooth_x(agg["steps"], smoothing)
            sy = smooth(arr, smoothing)
            ax.plot(sx, sy, color=COLORS[i % len(COLORS)], linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_title(TASK_TYPE_LABELS.get(tt, tt))
        ax.grid(True, alpha=0.3)
        _auto_ylim(ax)
        ax.set_ylabel("Mean Reward")

    fig.suptitle("Mean Reward by Task Type", fontsize=13)
    fig.tight_layout()
    return fig


def _collect_components_by_type(run_aggs):
    """Merge components_by_type across all runs."""
    merged = {}
    for _, agg in run_aggs:
        for tt, fields in agg.get("components_by_type", {}).items():
            if tt not in merged:
                merged[tt] = list(fields)
            else:
                for f in fields:
                    if f not in merged[tt]:
                        merged[tt].append(f)
    return merged


def plot_components_by_type(run_aggs, smoothing):
    """Grid: rows = task types, columns = components for that type. Lines = runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cbt = _collect_components_by_type(run_aggs)
    # Only plot types that have score components
    all_types = sorted(tt for tt, comps in cbt.items() if comps)

    if not all_types:
        return None

    max_cols = max(len(cbt[tt]) for tt in all_types)
    n_rows = len(all_types)

    fig, axes = plt.subplots(n_rows, max_cols, figsize=(5 * max_cols, 4 * n_rows), squeeze=False)

    for row, tt in enumerate(all_types):
        components = cbt[tt]
        for col in range(max_cols):
            ax = axes[row][col]
            if col >= len(components):
                ax.set_visible(False)
                continue

            f = components[col]
            key = f"mean_{f}"
            for i, (label, agg) in enumerate(run_aggs):
                if tt not in agg["by_type"] or key not in agg["by_type"][tt]:
                    continue
                arr = agg["by_type"][tt][key]
                if len(arr) == 0:
                    continue
                sx = smooth_x(agg["steps"], smoothing)
                sy = smooth(arr, smoothing)
                ax.plot(sx, sy, color=COLORS[i % len(COLORS)], linewidth=1.5)

            name = f.replace("_", " ").title()
            ax.set_title(f"{TASK_TYPE_LABELS.get(tt, tt)} - {name}")
            ax.set_xlabel("Training Step")
            ax.grid(True, alpha=0.3)
            _auto_ylim(ax)
            if col == 0:
                ax.set_ylabel("Mean Score")

    fig.suptitle("Reward Components by Task Type", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_components_std_by_type(run_aggs, smoothing):
    """Grid: rows = task types, columns = components. Shows per-prompt std (variance)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cbt = _collect_components_by_type(run_aggs)
    all_types = sorted(tt for tt, comps in cbt.items() if comps)

    if not all_types:
        return None

    max_cols = max(len(cbt[tt]) for tt in all_types)
    n_rows = len(all_types)

    fig, axes = plt.subplots(n_rows, max_cols, figsize=(5 * max_cols, 4 * n_rows), squeeze=False)

    for row, tt in enumerate(all_types):
        components = cbt[tt]
        for col in range(max_cols):
            ax = axes[row][col]
            if col >= len(components):
                ax.set_visible(False)
                continue

            f = components[col]
            key = f"std_{f}"
            for i, (label, agg) in enumerate(run_aggs):
                if tt not in agg["by_type"] or key not in agg["by_type"][tt]:
                    continue
                arr = agg["by_type"][tt][key]
                if len(arr) == 0:
                    continue
                sx = smooth_x(agg["steps"], smoothing)
                sy = smooth(arr, smoothing)
                ax.plot(sx, sy, color=COLORS[i % len(COLORS)], linewidth=1.5)

            name = f.replace("_", " ").title()
            ax.set_title(f"{TASK_TYPE_LABELS.get(tt, tt)} - {name} Std")
            ax.set_xlabel("Training Step")
            ax.grid(True, alpha=0.3)
            _auto_ylim(ax)
            if col == 0:
                ax.set_ylabel("Mean Per-Prompt Std")

    fig.suptitle("Reward Component Variance by Task Type", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_reward_distribution(run_aggs, run_step_data_map):
    """Overlaid histograms of reward values across all steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, agg) in enumerate(run_aggs):
        step_data = run_step_data_map[label]
        all_rewards = []
        for s in sorted(step_data.keys()):
            for tt in step_data[s]:
                all_rewards.append(step_data[s][tt]["rewards"])
        all_rewards = np.concatenate(all_rewards)
        ax.hist(all_rewards, bins=50, color=COLORS[i % len(COLORS)], alpha=0.4,
                edgecolor="white")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution (All Steps)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_task_type_distribution(run_aggs):
    """Show fraction of each task type over steps, one subplot per run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(run_aggs)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    # Generate colors for all task types using a colormap
    _all_tt = set()
    for _, agg in run_aggs:
        _all_tt.update(agg["by_type"].keys())
    _all_tt = sorted(_all_tt)
    cmap = plt.cm.get_cmap("tab20", max(len(_all_tt), 1))
    tt_colors = {tt: cmap(i) for i, tt in enumerate(_all_tt)}
    tt_colors["unknown"] = "#9e9e9e"

    for i, (label, agg) in enumerate(run_aggs):
        ax = axes[0][i]
        steps = agg["steps"]
        by_type = agg["by_type"]
        all_types = sorted(by_type.keys())

        fractions = {}
        for tt in all_types:
            fractions[tt] = by_type[tt]["count"] / np.maximum(
                sum(by_type[t]["count"] for t in all_types), 1)

        bottoms = np.zeros(len(steps))
        for tt in all_types:
            vals = fractions[tt]
            color = tt_colors.get(tt, "#9e9e9e")
            ax.bar(steps, vals, bottom=bottoms, width=1, color=color,
                   label=TASK_TYPE_LABELS.get(tt, tt), alpha=0.8)
            bottoms += vals

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Fraction")
        ax.set_title(label)
        ax.legend(fontsize=7)  # keep legend here since it shows task types, not runs
        ax.set_ylim(0, 1.05)

    fig.suptitle("Task Type Distribution Over Training", fontsize=13)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def make_summary_table(run_aggs, run_metrics_list=None):
    """Return a markdown summary table with per-task-type breakdown.

    `run_metrics_list` (optional) is a parallel list to `run_aggs`, each
    entry a (label, list[dict]) of train_metrics.jsonl rows. When provided,
    OPSD columns are added to the Overall block for SDAR-enabled runs.
    """
    lines = []

    # Build a label → rows map for SDAR metrics (if provided).
    metrics_by_label = {}
    if run_metrics_list is not None:
        for label, rows in run_metrics_list:
            metrics_by_label[label] = rows or []

    def _avg_opsd(label, key):
        rows = metrics_by_label.get(label, [])
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        try:
            arr = np.array(vals, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        return float(np.nanmean(arr)) if len(arr) else None

    any_sdar = any(
        any(k.startswith("opsd/") for k in (rows[0] if rows else {}))
        for _, rows in metrics_by_label.items()
    )

    # Overall table
    rows = []
    for label, agg in run_aggs:
        a = agg["all"]
        row = {
            "Run": label,
            "Steps": len(agg["steps"]),
            "Mean Reward": f"{np.nanmean(a['mean_reward']):.4f}",
            "Final Reward": f"{a['mean_reward'][-1]:.4f}" if len(a["mean_reward"]) > 0 else "N/A",
            "Mean Std": f"{np.nanmean(a['mean_std']):.4f}",
            "Zero-Var %": f"{100 * np.nanmean(a['frac_zero_var']):.1f}%",
        }
        if any_sdar:
            for col_name, key in (
                ("OPSD Loss (avg)", "opsd/loss"),
                ("Gate Mean (avg)", "opsd/gate_mean"),
                ("Gate Active Frac (avg)", "opsd/gate_active_frac"),
                ("Δ̄ (avg)", "opsd/teacher_minus_student_mean"),
            ):
                val = _avg_opsd(label, key)
                row[col_name] = (
                    f"{val:+.4f}" if (val is not None and key.endswith("_mean")
                                      and key == "opsd/teacher_minus_student_mean")
                    else (f"{val:.4f}" if val is not None else "—")
                )
        rows.append(row)

    if not rows:
        return "No runs selected."

    lines.append("### Overall")
    headers = list(rows[0].keys())
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")

    # Per task-type tables
    all_types = set()
    for _, agg in run_aggs:
        all_types.update(agg["task_types"])
    all_types = sorted(all_types)

    # Merge components_by_type across runs
    merged_cbt = _collect_components_by_type(run_aggs)

    for tt in all_types:
        components = merged_cbt.get(tt, [])
        tt_rows = []
        for label, agg in run_aggs:
            if tt not in agg["by_type"]:
                continue
            bt = agg["by_type"][tt]
            valid_mask = ~np.isnan(bt["mean_reward"])
            if not np.any(valid_mask):
                continue
            row = {
                "Run": label,
                "Samples": f"{int(np.nansum(bt['count']))}",
                "Mean Reward": f"{np.nanmean(bt['mean_reward']):.4f}",
            }
            for f in components:
                key = f"mean_{f}"
                if key in bt:
                    name = f.replace("_", " ").title()
                    row[name] = f"{np.nanmean(bt[key]):.4f}"
            tt_rows.append(row)

        if tt_rows:
            lines.append(f"\n### {TASK_TYPE_LABELS.get(tt, tt)}")
            headers = list(tt_rows[0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in tt_rows:
                lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rollout browser helpers
# ---------------------------------------------------------------------------

def _escape_html_tags(text):
    """Escape angle brackets so Gradio markdown doesn't swallow XML/HTML-like tags."""
    import re
    # Escape tags that aren't standard HTML (e.g. <exercise_description>, <format>, etc.)
    # but preserve <details>, <summary>, <br>, <b>, <i>, <em>, <strong>, <code>, <pre>
    safe_tags = {"details", "summary", "br", "b", "i", "em", "strong", "code", "pre", "hr",
                 "/details", "/summary", "/b", "/i", "/em", "/strong", "/code", "/pre"}
    def replace_tag(m):
        tag_content = m.group(1).split()[0].strip("/").lower()
        if tag_content in safe_tags or "/" + tag_content in safe_tags:
            return m.group(0)
        return m.group(0).replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"<([^>]+)>", replace_tag, text)


# === SDAR observability: per-token color renderer used by the
# "Rollouts (KL color)" tab. Takes the assistant response as a list of
# pre-decoded token strings and a parallel list of per-token signal values
# (KL or Δ_t), and renders one <span> per token with a viridis-style
# background colour proportional to the value. Per-token HTML escaping is
# required — token strings can contain `<`, `&`, etc.
def _diverging_hex(t: float) -> str:
    """Diverging signed colormap, t∈[-1,1] → CSS hex.

    t =  0 → white (no signal)
    t > 0  → yellow → red (warm — flagged direction)
    t < 0  → pale blue → deep blue (cool — opposite direction)

    For the OPSD KL view this gives:
        warm  ↔ D̂_RKL > 0  (student over-confident, SDAR gate ≈ 0, ignored)
        cool  ↔ D̂_RKL < 0  (student under-confident, SDAR gate ≈ 1, fired)
    """
    import math as _math
    t = 0.0 if _math.isnan(t) else max(-1.0, min(1.0, t))
    if t >= 0:
        stops = [
            (0.0, (255, 255, 255)),   # white
            (0.25, (255, 247, 188)),  # pale yellow
            (0.5, (254, 217, 100)),   # yellow
            (0.75, (244, 143, 79)),   # orange
            (1.0, (220, 50, 60)),     # red
        ]
    else:
        t = -t
        stops = [
            (0.0, (255, 255, 255)),   # white
            (0.25, (218, 232, 246)),  # very pale blue
            (0.5, (160, 200, 232)),   # pale blue
            (0.75, (88, 152, 211)),   # mid blue
            (1.0, (28, 95, 175)),     # deep blue
        ]
    for j in range(len(stops) - 1):
        a, ca = stops[j]
        b, cb = stops[j + 1]
        if t <= b:
            f = 0.0 if b == a else (t - a) / (b - a)
            r = int(ca[0] + (cb[0] - ca[0]) * f)
            g = int(ca[1] + (cb[1] - ca[1]) * f)
            bl = int(ca[2] + (cb[2] - ca[2]) * f)
            return f"#{r:02x}{g:02x}{bl:02x}"
    return "#dc323c"


def _fmt_signal_value(v):
    """Tooltip-friendly float formatting that doesn't collapse tiny values to 0."""
    av = abs(float(v))
    if av == 0:
        return "0"
    if av < 1e-3 or av >= 1e4:
        return f"{float(v):.3e}"
    return f"{float(v):.6f}"


def render_colored_response(tokens, values, vmin=None, vmax=None, signal_label="KL"):
    """Render the response as a sequence of <span>s tinted by per-token values.

    Coloring is signed: v=0 renders unhighlighted, v>0 fades yellow→red, v<0
    fades pale→deep blue. Saturation scales with |v| / max(|v|) over the
    sample. The hover tooltip shows the signed value.

    Args:
        tokens: list[str] decoded token strings (one per response token).
        values: list[float] same length, the signal per token.
        vmin, vmax: optional clip bounds on |value| (if None, derived from
            data as max(|v|)). Single |v|max value is used to scale both
            warm and cool sides symmetrically.
        signal_label: short label for hover tooltips.
    Returns a full HTML fragment string.
    """
    import html as _html
    if not tokens or not values or len(tokens) != len(values):
        return "<i>No per-token data for this sample.</i>"
    arr = np.array(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    abs_arr = np.abs(arr)
    abs_finite = abs_arr[np.isfinite(abs_arr)]
    raw_min = float(np.min(finite)) if len(finite) else 0.0
    raw_max = float(np.max(finite)) if len(finite) else 0.0
    raw_mean = float(np.mean(finite)) if len(finite) else 0.0
    nonzero_n = int(np.sum(finite != 0)) if len(finite) else 0
    # Symmetric scale: a single |v|max bounds both warm (v>0) and cool (v<0)
    # sides so the legend is symmetric and the hue purely reflects sign.
    abs_vmax = (
        float(vmax) if vmax is not None
        else (float(np.max(abs_finite)) if len(abs_finite) else 1.0)
    )
    abs_vmin = float(vmin) if vmin is not None else 0.0
    if abs_vmax <= abs_vmin:
        abs_vmax = abs_vmin + 1e-9
    parts = [
        '<div style="font-family: ui-monospace, SFMono-Regular, monospace; '
        'font-size: 13px; line-height: 1.7; white-space: pre-wrap; '
        'background: #ffffff; color: #1a1a1a; padding: 10px 12px; '
        'border: 1px solid #e0e0e0; border-radius: 6px;">'
    ]
    for tok, v in zip(tokens, arr):
        fv = float(v)
        mag = abs(fv)
        t_mag = (mag - abs_vmin) / (abs_vmax - abs_vmin)
        t_mag = 0.0 if not np.isfinite(t_mag) else max(0.0, min(1.0, t_mag))
        t_signed = t_mag if fv >= 0 else -t_mag
        # readable text: black almost everywhere, white only when the
        # background drops into the deepest red or deepest blue territory.
        text_color = "#fff" if t_mag > 0.85 else "#1a1a1a"
        esc = _html.escape(str(tok)).replace("\n", "↵\n")
        # At |t|≈0 just use no background — the parent's white shows through —
        # to make agreement positions visually disappear.
        if t_mag < 0.03:
            span_style = f"color:{text_color};padding:1px 2px;margin:0 1px;"
        else:
            color = _diverging_hex(t_signed)
            span_style = (
                f"background:{color};color:{text_color};"
                f"padding:1px 2px;margin:0 1px;border-radius:2px;"
            )
        parts.append(
            f'<span title="{signal_label}={_fmt_signal_value(fv)}" '
            f'style="{span_style}">{esc}</span>'
        )
    parts.append("</div>")
    # Signed legend: blue ← 0 → red, with the magnitude bound shown at each end.
    grad_stops = []
    for j in range(21):
        t = -1.0 + j * (2.0 / 20)
        grad_stops.append(f"{_diverging_hex(t)} {int((j / 20) * 100)}%")
    grad = ", ".join(grad_stops)
    parts.append(
        f'<div style="margin-top:8px;font-size:12px;color:#333;">'
        f'<b>{signal_label}</b> signed color range: '
        f'<code>−{_fmt_signal_value(abs_vmax)}</code> → '
        f'<code>0</code> → '
        f'<code>+{_fmt_signal_value(abs_vmax)}</code><br>'
        f'<span style="color:#555;">raw per-token stats (signed): '
        f'<code>min={_fmt_signal_value(raw_min)}</code>, '
        f'<code>max={_fmt_signal_value(raw_max)}</code>, '
        f'<code>mean={_fmt_signal_value(raw_mean)}</code>, '
        f'<code>nonzero={nonzero_n}/{len(arr)}</code></span><br>'
        f'<div style="display:inline-block;width:300px;height:14px;'
        f'background:linear-gradient(90deg,{grad});'
        f'border:1px solid #ccc;border-radius:3px;vertical-align:middle;'
        f'margin-top:2px;"></div>'
        f' <span style="vertical-align:middle;color:#555;">'
        f'v &lt; 0 (blue) ← agreement (white) → v &gt; 0 (red)</span></div>'
    )
    return "".join(parts)
# === END SDAR observability ===


# === SDAR observability: render the GT assistant response as a structured
# block (parsed [MOVEMENT ANALYSIS] / [SCORES] / [ERRORS] sections + the raw
# text). Used by the "Rollouts (KL color)" tab to show GT side-by-side with
# the colored rollout. Uses lightweight inline regex parsers to avoid pulling
# in nemo_rl from the dashboard process.
def _gt_extract_section(text, name):
    import re as _re
    if not text:
        return ""
    m = _re.search(
        rf"\[{name}\]\s*\n(.*?)(?=\[|$)", text, _re.IGNORECASE | _re.DOTALL
    )
    return m.group(1).strip() if m else ""


def _gt_extract_score(text, label):
    import re as _re
    if not text:
        return None
    scores = _gt_extract_section(text, "SCORES") or text
    m = _re.search(rf"{label}:\s*(\d+)", scores, _re.IGNORECASE)
    return int(m.group(1)) if m else None


def _gt_extract_errors(text):
    import re as _re
    if not text:
        return {}
    section = _gt_extract_section(text, "ERRORS")
    out = {}
    for line in section.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _re.match(r"^([^\d:]+?)\s*:\s*(\d+)\s*$", line)
        if m:
            out[m.group(1).strip()] = int(m.group(2))
    return out


def render_ground_truth(gt_text):
    """Render the raw GT response as a plain monospace block. Empty / missing
    GT renders a small placeholder."""
    import html as _html
    if not gt_text or not isinstance(gt_text, str):
        return "<i>No ground-truth available for this sample.</i>"
    return (
        '<div style="font-family: ui-sans-serif, -apple-system, system-ui, sans-serif; '
        'font-size: 13px; line-height: 1.5; background: #ffffff; '
        'border: 1px solid #d4d4d4; border-radius: 6px; padding: 10px 12px; '
        'color: #111111;">'
        '<div style="font-weight: 700; font-size: 13px; margin-bottom: 8px; '
        'color: #111111; border-bottom: 1px solid #ececec; padding-bottom: 4px;">'
        'Ground truth</div>'
        '<pre style="white-space:pre-wrap;background:#fafafa;color:#111111;'
        'padding:8px;border:1px solid #d4d4d4;border-radius:4px;margin:0;'
        'font-size:12px;font-family: ui-monospace, SFMono-Regular, monospace;">'
        f"{_html.escape(gt_text)}</pre>"
        '</div>'
    )
# === END SDAR observability ===


def _extract_user_prompt(content):
    """Extract the full user prompt from a content list."""
    parts = []
    if len(content) > 1 and isinstance(content[1], list):
        for item in content[1]:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item["text"])
            elif isinstance(item, dict) and item.get("type") == "video":
                parts.append(f"[Video: {item.get('count', '?')} frames]")
    return "\n\n".join(parts)


def _get_task_type(rec):
    """Get task type from a record, inferring from available scores if needed."""
    # New format: check reward_details first
    rd = rec.get("reward_details")
    if rd and isinstance(rd, list) and rd[0] and isinstance(rd[0], dict):
        tt = _infer_task_type_from_details(rd[0])
        if tt and tt != "unknown":
            return tt
    # Top-level task_type field
    if "task_type" in rec:
        tt = rec["task_type"]
        tt = tt[0] if isinstance(tt, list) else tt
        if tt:
            return tt
    # Old format inference: detection → repetition, correctness → full_exercise
    det = rec.get("detection_score", [None])[0]
    corr = rec.get("correctness_score", [None])[0]
    if det is not None:
        return "repetition"
    elif corr is not None:
        return "full_exercise"
    return "unknown"


def group_records_by_sample(records):
    """Group records by sample_id, preserving insertion order."""
    groups = OrderedDict()
    for rec in records:
        sid = rec.get("sample_id", [None])[0]
        groups.setdefault(sid, []).append(rec)
    return list(groups.items())


def format_sample_group(sample_id, rollouts, group_idx):
    """Format all rollouts for a single sample, with advantages."""
    rewards = [r["rewards"][0] for r in rollouts]
    mean_reward = np.mean(rewards)
    advantages = [rw - mean_reward for rw in rewards]
    task_type = _get_task_type(rollouts[0])

    parts = [f"## Sample {group_idx}: `{sample_id}`"]
    parts.append(f"**Task Type:** `{task_type}` | **Rollouts:** {len(rollouts)} | "
                 f"**Mean Reward:** {mean_reward:.4f} | "
                 f"**Min:** {min(rewards):.4f} | **Max:** {max(rewards):.4f} | "
                 f"**Std:** {np.std(rewards, ddof=1) if len(rewards) > 1 else 0:.4f}")

    # Shared user prompt
    content = rollouts[0].get("content", [[]])[0]
    user_input = _extract_user_prompt(content)
    if user_input:
        parts.append(f"\n<details><summary>User Prompt (click to expand)</summary>\n\n{_escape_html_tags(user_input)}\n\n</details>")

    # Each rollout
    for j, rec in enumerate(rollouts):
        content = rec.get("content", [[]])[0]
        model_response = content[2] if len(content) > 2 else "(no response)"
        reward = rewards[j]
        advantage = advantages[j]

        parts.append(f"\n### Rollout {j + 1}")

        adv_sign = "+" if advantage >= 0 else ""
        parts.append(f"**Reward:** `{reward:.4f}` | **Advantage:** `{adv_sign}{advantage:.4f}`")

        # Extract scores and extra fields from reward_details (new) or top-level (old)
        rd = rec.get("reward_details")
        if rd and isinstance(rd, list) and rd[0] and isinstance(rd[0], dict):
            details = rd[0]
            components = get_task_components(task_type)
            if not components:
                components = [
                    k for k, v in details.items()
                    if isinstance(v, (int, float)) and k not in _NON_SCORE_KEYS
                ]
            score_parts = []
            for f in components:
                val = details.get(f)
                if val is not None:
                    score_parts.append(f"{f.replace('_', ' ').title()}: {val}")
            if score_parts:
                parts.append(f"**Scores:** {' | '.join(score_parts)}")

            extra_fields = EXTRA_DETAIL_FIELDS.get(task_type, [])
            extra_parts = []
            for f in extra_fields:
                val = details.get(f)
                if val is not None:
                    extra_parts.append(f"{f}: {val}")
            if extra_parts:
                parts.append(f"**Extra:** {' | '.join(extra_parts)}")
        else:
            # Old format: scores at top level with _score suffix
            components = get_task_components(task_type)
            if not components:
                components = list(OLD_FIELD_MAP.values())
            score_parts = []
            for f in components:
                old_name = f + "_score"
                val = None
                if old_name in rec:
                    val = rec[old_name]
                    val = val[0] if isinstance(val, list) else val
                elif f in rec:
                    val = rec[f]
                    val = val[0] if isinstance(val, list) else val
                if val is not None:
                    score_parts.append(f"{f.replace('_', ' ').title()}: {val}")
            if score_parts:
                parts.append(f"**Scores:** {' | '.join(score_parts)}")

            extra_fields = EXTRA_DETAIL_FIELDS.get(task_type, [])
            extra_parts = []
            for f in extra_fields:
                if f in rec:
                    val = rec[f]
                    val = val[0] if isinstance(val, list) else val
                    if val is not None:
                        extra_parts.append(f"{f}: {val}")
            if extra_parts:
                parts.append(f"**Extra:** {' | '.join(extra_parts)}")

        # Model response
        parts.append(f"\n<details><summary>Model Response (click to expand)</summary>\n\n{_escape_html_tags(model_response)}\n\n</details>")

    parts.append("\n---\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# App state & caching
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self, logs_dir):
        self.logs_dir = logs_dir
        self.runs = discover_runs(logs_dir)
        self.run_map = {name: path for name, path in self.runs}
        self.short_names = _build_short_name_map([name for name, _ in self.runs])
        self._step_data_cache = {}
        self._agg_cache = {}
        # === SDAR observability: cache for train_metrics.jsonl rows per run.
        self._train_metrics_cache = {}

    def run_names(self):
        return [name for name, _ in self.runs]

    def short_name(self, full_name):
        return self.short_names.get(full_name, shorten_run_name(full_name))

    def get_step_data(self, run_name):
        if run_name not in self._step_data_cache:
            path = self.run_map[run_name]
            self._step_data_cache[run_name] = load_run_metrics(path, run_name=run_name)
        return self._step_data_cache[run_name]

    def get_agg(self, run_name, num_rollouts=8):
        key = (run_name, num_rollouts)
        if key not in self._agg_cache:
            step_data = self.get_step_data(run_name)
            self._agg_cache[key] = aggregate_run(step_data, num_rollouts)
        return self._agg_cache[key]

    # === SDAR observability: lazy-loaded per-run train_metrics.jsonl rows.
    def get_train_metrics(self, run_name):
        """Return list[dict] of per-step rows from train_metrics.jsonl for
        the run (empty list if the run is GRPO-only / has no such file)."""
        if run_name not in self._train_metrics_cache:
            path = self.run_map.get(run_name)
            if path is None:
                self._train_metrics_cache[run_name] = []
            else:
                self._train_metrics_cache[run_name] = load_run_train_metrics(path)
        return self._train_metrics_cache[run_name]

    def get_steps_for_run(self, run_name):
        paths = self.run_map.get(run_name)
        if not paths:
            return []
        if isinstance(paths, str):
            paths = [paths]
        steps = set()
        for path in paths:
            for fp in glob.glob(os.path.join(path, "train_data_step*.jsonl")):
                m = re.search(r"step(\d+)", os.path.basename(fp))
                if m:
                    steps.add(int(m.group(1)))
        return sorted(steps)

    def get_task_types_for_run(self, run_name):
        """Get all task types found across all steps of a run."""
        step_data = self.get_step_data(run_name)
        if not step_data:
            return []
        all_types = set()
        for step_dict in step_data.values():
            all_types.update(step_dict.keys())
        return sorted(all_types)


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def create_app(logs_dir):
    state = AppState(logs_dir)
    run_names = state.run_names()

    with gr.Blocks(title="GRPO Training Dashboard", theme=gr.themes.Soft()) as app:
        gr.Markdown("# GRPO Training Dashboard\nCompare runs, visualize reward metrics, and browse rollouts.")

        # ===== Tab 1: Compare Runs =====
        with gr.Tab("Compare Runs"):
            with gr.Row():
                run_selector = gr.Dropdown(
                    choices=run_names, multiselect=True, label="Select Runs to Compare",
                    info="Pick one or more runs to overlay",
                )
                num_rollouts_input = gr.Number(value=8, label="Rollouts per Prompt", precision=0)
                smoothing_input = gr.Slider(1, 200, value=10, step=1, label="Smoothing Window")

            load_btn = gr.Button("Load & Compare", variant="primary")

            summary_md = gr.Markdown(label="Summary Table")
            legend_plot = gr.Plot(label="Legend")

            with gr.Row():
                reward_plot = gr.Plot(label="Mean Reward")
                std_plot = gr.Plot(label="Reward Diversity")

            with gr.Row():
                zero_var_plot = gr.Plot(label="Zero-Variance Fraction")
                dist_plot = gr.Plot(label="Reward Distribution")

            reward_by_type_plot = gr.Plot(label="Mean Reward by Task Type")
            comp_by_type_plot = gr.Plot(label="Components by Task Type")
            comp_std_by_type_plot = gr.Plot(label="Component Variance by Task Type")
            task_dist_plot = gr.Plot(label="Task Type Distribution")

            # === SDAR observability: four extra plots, shown when any of the
            # selected runs has a train_metrics.jsonl with opsd/* keys.
            gr.Markdown("---\n#### SDAR / OPSD (only populated for runs with `train_metrics.jsonl`)")
            with gr.Row():
                opsd_loss_plot = gr.Plot(label="OPSD Loss")
                opsd_gate_mean_plot = gr.Plot(label="OPSD Gate Mean")
            with gr.Row():
                opsd_gate_active_plot = gr.Plot(label="OPSD Gate Active Fraction")
                opsd_delta_plot = gr.Plot(label="Teacher − Student Gap Δ̄")
            # === END SDAR observability ===

            def on_compare(selected_runs, num_rollouts, smoothing):
                if not selected_runs:
                    empty = "No runs selected."
                    return (
                        empty, None, None, None, None, None, None, None, None, None,
                        None, None, None, None,
                    )

                num_rollouts = int(num_rollouts)
                smoothing = int(smoothing)

                run_aggs = []
                run_step_data_map = {}
                run_metrics_list = []
                for rn in selected_runs:
                    short = state.short_name(rn)
                    agg = state.get_agg(rn, num_rollouts)
                    run_aggs.append((short, agg))
                    run_step_data_map[short] = state.get_step_data(rn)
                    run_metrics_list.append((short, state.get_train_metrics(rn)))

                summary = make_summary_table(run_aggs, run_metrics_list=run_metrics_list)
                fig_legend = make_legend_figure(run_aggs)
                fig_reward = plot_mean_reward(run_aggs, smoothing)
                fig_std = plot_mean_std(run_aggs, smoothing)
                fig_zv = plot_zero_var(run_aggs, smoothing)
                fig_dist = plot_reward_distribution(run_aggs, run_step_data_map)
                fig_reward_tt = plot_mean_reward_by_type(run_aggs, smoothing)
                fig_comp_tt = plot_components_by_type(run_aggs, smoothing)
                fig_comp_std_tt = plot_components_std_by_type(run_aggs, smoothing)
                fig_task_dist = plot_task_type_distribution(run_aggs)

                # SDAR plots — silently skip if no run has SDAR data.
                fig_opsd_loss = plot_opsd_loss(run_metrics_list, smoothing)
                fig_opsd_gate_mean = plot_opsd_gate_mean(run_metrics_list, smoothing)
                fig_opsd_gate_active = plot_opsd_gate_active_frac(run_metrics_list, smoothing)
                fig_opsd_delta = plot_opsd_teacher_minus_student(run_metrics_list, smoothing)

                return (summary, fig_legend, fig_reward, fig_std, fig_zv, fig_dist,
                        fig_reward_tt, fig_comp_tt, fig_comp_std_tt, fig_task_dist,
                        fig_opsd_loss, fig_opsd_gate_mean, fig_opsd_gate_active, fig_opsd_delta)

            load_btn.click(
                fn=on_compare,
                inputs=[run_selector, num_rollouts_input, smoothing_input],
                outputs=[summary_md, legend_plot, reward_plot, std_plot, zero_var_plot, dist_plot,
                         reward_by_type_plot, comp_by_type_plot, comp_std_by_type_plot, task_dist_plot,
                         opsd_loss_plot, opsd_gate_mean_plot, opsd_gate_active_plot, opsd_delta_plot],
            )

        # ===== Tab 2: Rollout Browser =====
        with gr.Tab("Rollout Browser"):
            with gr.Row():
                browse_run = gr.Dropdown(choices=run_names, label="Run", info="Select a run")
                browse_step = gr.Dropdown(choices=[], label="Step", info="Select a step")
                browse_task_type = gr.Dropdown(
                    choices=["all"], value="all", label="Task Type",
                    info="Filter by task type (populated when run is selected)"
                )

            with gr.Row():
                filter_reward_min = gr.Number(value=0.0, label="Min Mean Reward")
                filter_reward_max = gr.Number(value=1.0, label="Max Mean Reward")
                page_num = gr.Number(value=1, label="Page (samples)", precision=0)
                page_size = gr.Number(value=5, label="Samples per Page", precision=0)

            browse_btn = gr.Button("Load Rollouts", variant="primary")
            rollout_info = gr.Markdown()
            rollout_display = gr.Markdown()

            def update_steps_and_types(run_name):
                if not run_name:
                    return gr.update(choices=[], value=None), gr.update(choices=["all"], value="all")
                steps = state.get_steps_for_run(run_name)
                step_strs = [str(s) for s in steps]
                task_types = state.get_task_types_for_run(run_name)
                return (
                    gr.update(choices=step_strs, value=step_strs[0] if step_strs else None),
                    gr.update(choices=["all"] + task_types, value="all"),
                )

            browse_run.change(fn=update_steps_and_types, inputs=[browse_run], outputs=[browse_step, browse_task_type])

            def on_browse(run_name, step_str, task_type_filter, rmin, rmax, page, psize):
                if not run_name or not step_str:
                    return "Select a run and step.", ""

                step = int(step_str)
                page = max(1, int(page))
                psize = max(1, int(psize))
                path = state.run_map[run_name]
                records = load_step_rollouts(path, step)

                # Filter by task type
                if task_type_filter and task_type_filter != "all":
                    records = [r for r in records if _get_task_type(r) == task_type_filter]

                # Group by sample_id
                sample_groups = group_records_by_sample(records)

                # Filter by mean reward
                filtered = []
                for sid, rollouts in sample_groups:
                    mean_rw = np.mean([r["rewards"][0] for r in rollouts])
                    if rmin <= mean_rw <= rmax:
                        filtered.append((sid, rollouts))

                total = len(filtered)
                start = (page - 1) * psize
                end = min(start + psize, total)
                page_groups = filtered[start:end]

                total_rollouts = sum(len(rols) for _, rols in filtered)
                tt_info = f" (task_type={task_type_filter})" if task_type_filter != "all" else ""
                info = (f"**{total}** samples ({total_rollouts} rollouts) match{tt_info} "
                        f"(mean reward in [{rmin}, {rmax}]). "
                        f"Showing samples {start+1}-{end} (page {page}).")

                output_parts = []
                for i, (sid, rollouts) in enumerate(page_groups):
                    output_parts.append(format_sample_group(sid, rollouts, start + i + 1))

                return info, "\n".join(output_parts) if output_parts else "No samples on this page."

            browse_btn.click(
                fn=on_browse,
                inputs=[browse_run, browse_step, browse_task_type,
                        filter_reward_min, filter_reward_max, page_num, page_size],
                outputs=[rollout_info, rollout_display],
            )

        # ===== Tab 3: Rollouts (KL color) =====
        # === SDAR observability: per-token KL/Δ-coloured rendering of the
        # assistant response. Reads response_tokens + response_per_token_kl
        # (and optionally response_per_token_delta) from train_data_step{N}.jsonl
        # — keys emitted by sdar.async_sdar_train when SDAR is active.
        with gr.Tab("Rollouts (KL color)"):
            gr.Markdown(
                "Pick a run / step / sample and view the assistant response "
                "with each token shaded by its per-token signal."
            )
            with gr.Row():
                kl_run = gr.Dropdown(choices=run_names, label="Run", info="SDAR run with per-token KL data")
                kl_step = gr.Dropdown(choices=[], label="Step")
                kl_signal = gr.Dropdown(
                    choices=[
                        "KL (policy ‖ reference)",
                        "Δ_t (teacher − student)",
                    ],
                    value="KL (policy ‖ reference)",
                    label="Coloring signal",
                )
            with gr.Row():
                kl_sample_idx = gr.Number(value=0, label="Sample index (0-based)", precision=0)

            kl_load_btn = gr.Button("Render", variant="primary")
            kl_info = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("**Assistant rollout — colored by signal**")
                    kl_render = gr.HTML()
                with gr.Column(scale=1):
                    gr.Markdown("**Ground truth**")
                    kl_gt_render = gr.HTML()

            def _kl_update_steps(run_name):
                if not run_name:
                    return gr.update(choices=[], value=None)
                steps = state.get_steps_for_run(run_name)
                step_strs = [str(s) for s in steps]
                return gr.update(
                    choices=step_strs,
                    value=step_strs[0] if step_strs else None,
                )

            kl_run.change(fn=_kl_update_steps, inputs=[kl_run], outputs=[kl_step])

            def on_kl_render(run_name, step_str, signal_choice, sample_idx):
                if not run_name or not step_str:
                    return "Select a run and step.", "", ""
                step = int(step_str)
                path = state.run_map[run_name]
                records = load_step_rollouts(path, step)
                if not records:
                    return (
                        f"No rollouts found for step {step}.",
                        "<i>No data.</i>",
                        "<i>No data.</i>",
                    )
                sample_idx = max(0, int(sample_idx))
                if sample_idx >= len(records):
                    sample_idx = len(records) - 1
                rec = records[sample_idx]

                if "Δ" in signal_choice or "delta" in signal_choice.lower():
                    key = "response_per_token_delta"
                    signal_label = "Δ_t (teacher − student)"
                else:
                    key = "response_per_token_kl"
                    signal_label = "KL"

                tokens = rec.get("response_tokens", [])
                # Per-sample arrays: when log_batched_dict_as_jsonl iterates
                # microbatches of size 1, fields are unwrapped to scalars,
                # but list-of-list values come through as the inner list.
                if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
                    tokens = tokens[0]
                values = rec.get(key, [])
                if isinstance(values, list) and len(values) > 0 and isinstance(values[0], list):
                    values = values[0]

                # Ground-truth — same nested-list unwrap pattern.
                gt_text = rec.get("ground_truth", "")
                if isinstance(gt_text, list):
                    gt_text = gt_text[0] if gt_text else ""

                if not tokens or not values:
                    return (
                        f"Step {step}, sample {sample_idx}: no `response_tokens` / "
                        f"`{key}` in this rollout. Make sure the run was launched "
                        f"with SDAR (`log_per_token_decoded` defaults to true).",
                        "<i>No per-token data available for this sample.</i>",
                        render_ground_truth(gt_text),
                    )

                reward = rec.get("rewards", [None])
                reward_val = reward[0] if isinstance(reward, list) and reward else reward
                sid = rec.get("sample_id", [None])
                if isinstance(sid, list):
                    sid = sid[0] if sid else None
                tt = _get_task_type(rec)

                info = (
                    f"**Step {step}**, sample index `{sample_idx}` (of {len(records)})  \n"
                    f"`sample_id={sid}` · `task_type={tt}` · `reward={reward_val}`  \n"
                    f"Response length: **{len(tokens)} tokens**. Signal: **{signal_label}**."
                )
                # vmin / vmax = None → render_colored_response anchors |v|=0 to
                # white and saturates at max(|v|) for the sample. No clipping.
                html = render_colored_response(
                    tokens, values, vmin=None, vmax=None, signal_label=signal_label
                )
                gt_html = render_ground_truth(gt_text)
                return info, html, gt_html

            kl_load_btn.click(
                fn=on_kl_render,
                inputs=[kl_run, kl_step, kl_signal, kl_sample_idx],
                outputs=[kl_info, kl_render, kl_gt_render],
            )
        # === END SDAR observability ===

        # ===== Tab 4: Single Run Deep Dive =====
        with gr.Tab("Single Run Analysis"):
            with gr.Row():
                single_run = gr.Dropdown(choices=run_names, label="Run")
                single_rollouts = gr.Number(value=8, label="Rollouts per Prompt", precision=0)
                single_smoothing = gr.Slider(1, 200, value=10, step=1, label="Smoothing Window")

            single_btn = gr.Button("Analyze", variant="primary")

            single_summary = gr.Markdown()
            single_legend_plot = gr.Plot(label="Legend")
            with gr.Row():
                single_reward_plot = gr.Plot(label="Mean Reward")
                single_std_plot = gr.Plot(label="Reward Diversity")
            with gr.Row():
                single_zv_plot = gr.Plot(label="Zero-Variance Fraction")
                single_dist_plot = gr.Plot(label="Reward Distribution")
            single_reward_tt_plot = gr.Plot(label="Mean Reward by Task Type")
            single_comp_tt_plot = gr.Plot(label="Components by Task Type")
            single_comp_std_tt_plot = gr.Plot(label="Component Variance by Task Type")

            # === SDAR observability: SDAR plots also exposed on Single Run.
            gr.Markdown("---\n#### SDAR / OPSD")
            with gr.Row():
                single_opsd_loss_plot = gr.Plot(label="OPSD Loss")
                single_opsd_gate_mean_plot = gr.Plot(label="OPSD Gate Mean")
            with gr.Row():
                single_opsd_gate_active_plot = gr.Plot(label="OPSD Gate Active Fraction")
                single_opsd_delta_plot = gr.Plot(label="Teacher − Student Gap Δ̄")
            # === END SDAR observability ===

            def on_single(run_name, num_rollouts, smoothing):
                if not run_name:
                    return "No run selected.", *([None] * 12)

                num_rollouts = int(num_rollouts)
                smoothing = int(smoothing)

                short = state.short_name(run_name)
                agg = state.get_agg(run_name, num_rollouts)
                run_aggs = [(short, agg)]
                run_step_data_map = {short: state.get_step_data(run_name)}
                run_metrics_list = [(short, state.get_train_metrics(run_name))]

                summary = make_summary_table(run_aggs, run_metrics_list=run_metrics_list)
                fig_legend = make_legend_figure(run_aggs)
                fig_reward = plot_mean_reward(run_aggs, smoothing)
                fig_std = plot_mean_std(run_aggs, smoothing)
                fig_zv = plot_zero_var(run_aggs, smoothing)
                fig_dist = plot_reward_distribution(run_aggs, run_step_data_map)
                fig_reward_tt = plot_mean_reward_by_type(run_aggs, smoothing)
                fig_comp_tt = plot_components_by_type(run_aggs, smoothing)
                fig_comp_std_tt = plot_components_std_by_type(run_aggs, smoothing)

                fig_opsd_loss = plot_opsd_loss(run_metrics_list, smoothing)
                fig_opsd_gate_mean = plot_opsd_gate_mean(run_metrics_list, smoothing)
                fig_opsd_gate_active = plot_opsd_gate_active_frac(run_metrics_list, smoothing)
                fig_opsd_delta = plot_opsd_teacher_minus_student(run_metrics_list, smoothing)

                return (summary, fig_legend, fig_reward, fig_std, fig_zv, fig_dist,
                        fig_reward_tt, fig_comp_tt, fig_comp_std_tt,
                        fig_opsd_loss, fig_opsd_gate_mean, fig_opsd_gate_active, fig_opsd_delta)

            single_btn.click(
                fn=on_single,
                inputs=[single_run, single_rollouts, single_smoothing],
                outputs=[single_summary, single_legend_plot, single_reward_plot, single_std_plot,
                         single_zv_plot, single_dist_plot, single_reward_tt_plot, single_comp_tt_plot,
                         single_comp_std_tt_plot,
                         single_opsd_loss_plot, single_opsd_gate_mean_plot,
                         single_opsd_gate_active_plot, single_opsd_delta_plot],
            )

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO Training Dashboard")
    parser.add_argument("--logs-dir", default=DEFAULT_LOGS_DIR, help="Root directory containing run folders")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    app = create_app(args.logs_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
