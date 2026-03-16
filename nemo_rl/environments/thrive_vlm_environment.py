# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import io
import json
import logging
import re
from typing import Any, Optional, TypedDict

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, RayVirtualCluster
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.judge_config import JudgeConfig
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers
from nemo_rl.models.generation.vllm import VllmGeneration


class ThriveVLMEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env
    reward_mode: Optional[str]  # "vanilla", "vanilla_quadratic", "vanilla_sqrt", "weighted", "weighted_quadratic", "weighted_sqrt", "weighted_separate_norm", "weighted_separate_norm_quadratic", "weighted_separate_norm_sqrt", "two_stage", "two_stage_sqrt", "two_stage_weighted", "detection_correctness_severity", "detection_correctness_severity_quadratic", "detection_correctness_severity_sqrt" (default: "vanilla")
    # Weighted mode parameters
    error_weight: Optional[float]  # Weight for errors with GT severity > 1 or movement < 6 (default: 1.0)
    non_error_weight: Optional[float]  # Weight for errors with GT severity = 1 or movement = 6 (default: 0.5)
    # Two-stage mode parameters
    detection_weight: Optional[float]  # Weight for detection accuracy (default: 0.3)
    severity_weight: Optional[float]  # Weight for severity accuracy (default: 0.7)
    # Detection-Correctness-Severity mode parameters
    correctness_weight: Optional[float]  # Weight for exact correctness (default: 0.2)
    # Full-exercise analysis reward mode ("vanilla", "vanilla_quadratic", "vanilla_sqrt"; default: "vanilla")
    full_exercise_reward_mode: Optional[str]
    # Judge configuration - when enabled, LLM judge is used instead of rule-based verification
    # The judge can work with any reward_mode to determine how to evaluate and score responses
    judge: Optional[JudgeConfig]


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


def normalize_error_name(name: str) -> str:
    """Normalize error name for matching (case-insensitive, ignore punctuation)."""
    return re.sub(r'[^a-z0-9]', '', name.lower())


# Ordinal index mappings for Q1–Q9 in the Full Exercise Analysis format.
# Keys are lowercase answer strings; values are 0-based ordinal positions (best → worst).
Q_ORDINALS: dict[int, dict[str, int]] = {
    # Q1 — Consistency (ROM, alignment, control)
    0: {
        "highly consistent": 0,
        "mostly consistent": 1,
        "moderate variability": 2,
        "high variability": 3,
    },
    # Q2 — Control and Stability
    1: {
        "excellent": 0,
        "good": 1,
        "moderate": 2,
        "poor": 3,
    },
    # Q3 — Joint Alignment
    2: {
        "consistently proper": 0,
        "mostly proper": 1,
        "mostly poor": 2,
        "consistently poor": 3,
    },
    # Q4 — Range of Motion
    3: {
        "full/optimal": 0,
        "slightly reduced": 1,
        "moderately reduced": 2,
        "significantly reduced": 3,
    },
    # Q5 — Errors (frequency/severity)
    4: {
        "none": 0,
        "few minor": 1,
        "several minor": 2,
        "few major": 3,
        "frequent and/or severe": 4,
    },
    # Q6 — Pacing
    5: {
        "smooth and controlled": 0,
        "mostly consistent": 1,
        "some inconsistency": 2,
        "erratic": 3,
    },
    # Q7 — Symmetry
    6: {
        "fully symmetrical": 0,
        "minor asymmetry": 1,
        "noticeable asymmetry": 2,
        "significant asymmetry": 3,
    },
    # Q8 — Fatigue Signs
    7: {
        "none": 0,
        "mild in final reps": 1,
        "moderate": 2,
        "significant": 3,
    },
    # Q9 — Trunk Posture
    8: {
        "excellent": 0,
        "good": 1,
        "moderate issues": 2,
        "poor": 3,
    },
}


@ray.remote
class ThriveVLMVerifyWorker:
    def __init__(self, cfg: ThriveVLMEnvConfig) -> None:
        logging.getLogger("thrive_vlm_worker").setLevel(logging.CRITICAL)

        # Reward mode configuration (per-rep task)
        self.reward_mode = cfg.get("reward_mode", "vanilla")

        # Weighted mode parameters
        self.error_weight = cfg.get("error_weight", 1.0)
        self.non_error_weight = cfg.get("non_error_weight", 0.5)

        # Two-stage mode parameters
        self.detection_weight = cfg.get("detection_weight", 0.3)
        self.severity_weight = cfg.get("severity_weight", 0.7)

        # Detection-Correctness-Severity mode parameters
        self.correctness_weight = cfg.get("correctness_weight", 0.2)

        # Full-exercise analysis reward mode
        self.full_exercise_reward_mode = cfg.get("full_exercise_reward_mode", "vanilla")

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        task_types: Optional[list[str]] = None,
    ) -> list[float]:
        """Verify the correctness of the predicted responses against the ground truth.

        Computes distance-based rewards for severity scores and movement scores.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. Ground truth text in the same format as responses.
            task_types: Optional per-sample task type. Each element is either
                ``"repetition"`` (per-rep severity reward) or ``"full_exercise"``
                (Q1–Q9 ordinal reward).  Defaults to all ``"repetition"``.

        Returns:
            list[float]. The rewards for each predicted response.
        """
        if task_types is None:
            task_types = ["repetition"] * len(pred_responses)

        results = []
        for idx, (response, ground_truth_str, task_type) in enumerate(zip(pred_responses, ground_truths, task_types)):
            try:
                reward = self._compute_reward(response, ground_truth_str, task_type=task_type, sample_idx=idx)
                results.append(float(reward))
            except Exception:
                results.append(0.0)
        return results

    def _compute_reward(
        self, response: str, ground_truth_str: str, task_type: str = "repetition", sample_idx: int = -1
    ) -> float:
        """Compute distance-based reward for thrive-vlm responses.

        Args:
            response: Model's generated response
            ground_truth_str: Ground truth text (same format as response)
            task_type: ``"repetition"`` uses per-error severity reward;
                       ``"full_exercise"`` uses Q1–Q9 ordinal reward.

        Returns:
            float: Reward between 0.0 and 1.0
        """
        if task_type == "full_exercise":
            # Resolve distance type from full_exercise_reward_mode
            mode = self.full_exercise_reward_mode or "vanilla"
            if mode.endswith("_quadratic"):
                distance_type = "quadratic"
            elif mode.endswith("_sqrt"):
                distance_type = "sqrt"
            else:
                distance_type = "linear"
            return self._compute_full_exercise_reward(
                response, ground_truth_str, distance_type=distance_type
            )

        # --- Per-repetition severity reward (original logic) ---
        # Parse ground truth scores from text (same format as response)
        gt_movement_score = self._extract_movement_score(ground_truth_str)
        gt_injury_risk = self._extract_injury_risk(ground_truth_str)
        gt_errors = self._extract_error_scores(ground_truth_str)

        if gt_movement_score is None:
            print(f"❌ Ground truth missing movement_score. Ground truth text: {ground_truth_str[:200]}", flush=True)
            return 0.0

        # Parse predicted scores from response
        pred_movement_score = self._extract_movement_score(response)
        pred_injury_risk = self._extract_injury_risk(response)
        pred_errors = self._extract_error_scores(response)

        if pred_movement_score is None:
            print(f"❌ Failed to extract movement score from response: {response[:300]}", flush=True)
            return 0.0

        # DEBUG: Print parsed scores for first 2 samples every step
        if sample_idx < 2:
            print("\n" + "="*80)
            print(f"PARSED SCORES [Sample {sample_idx}]:")
            print(f"\nPredicted:")
            print(f"  Effectiveness: {pred_movement_score}")
            print(f"  Injury Risk: {pred_injury_risk}")
            print(f"  Errors: {pred_errors}")
            print(f"\nGround Truth:")
            print(f"  Effectiveness: {gt_movement_score}")
            print(f"  Injury Risk: {gt_injury_risk}")
            print(f"  Errors: {gt_errors}")
            print("="*80 + "\n")

        # Compute base severity reward (distance-based)
        severity_reward = self._compute_severity_reward(
            gt_movement_score, gt_errors, pred_movement_score, pred_errors,
            gt_injury_risk=gt_injury_risk, pred_injury_risk=pred_injury_risk
        )

        # Apply reward mode
        if self.reward_mode == "vanilla":
            final_reward = severity_reward

        elif self.reward_mode == "vanilla_quadratic":
            final_reward = self._compute_severity_reward(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="quadratic", gt_injury_risk=gt_injury_risk, pred_injury_risk=pred_injury_risk
            )

        elif self.reward_mode == "vanilla_sqrt":
            final_reward = self._compute_severity_reward(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="sqrt", gt_injury_risk=gt_injury_risk, pred_injury_risk=pred_injury_risk
            )

        elif self.reward_mode == "weighted":
            final_reward = self._compute_severity_reward_weighted(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="linear", gt_injury_risk=gt_injury_risk, pred_injury_risk=pred_injury_risk
            )

        elif self.reward_mode == "weighted_quadratic":
            final_reward = self._compute_severity_reward_weighted(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="quadratic", gt_injury_risk=gt_injury_risk, pred_injury_risk=pred_injury_risk
            )

        elif self.reward_mode == "weighted_sqrt":
            final_reward = self._compute_severity_reward_weighted(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="sqrt", gt_injury_risk=gt_injury_risk, pred_injury_risk=pred_injury_risk
            )

        elif self.reward_mode == "weighted_separate_norm":
            if gt_injury_risk is None or pred_injury_risk is None:
                print("⚠️  weighted_separate_norm mode requires injury_risk scores, falling back to weighted", flush=True)
                final_reward = self._compute_severity_reward_weighted(
                    gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                    distance_type="linear"
                )
            else:
                final_reward = self._compute_severity_reward_weighted_separate_norm(
                    gt_movement_score, gt_injury_risk, gt_errors,
                    pred_movement_score, pred_injury_risk, pred_errors,
                    distance_type="linear"
                )

        elif self.reward_mode == "weighted_separate_norm_quadratic":
            if gt_injury_risk is None or pred_injury_risk is None:
                print("⚠️  weighted_separate_norm_quadratic mode requires injury_risk scores, falling back to weighted_quadratic", flush=True)
                final_reward = self._compute_severity_reward_weighted(
                    gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                    distance_type="quadratic"
                )
            else:
                final_reward = self._compute_severity_reward_weighted_separate_norm(
                    gt_movement_score, gt_injury_risk, gt_errors,
                    pred_movement_score, pred_injury_risk, pred_errors,
                    distance_type="quadratic"
                )

        elif self.reward_mode == "weighted_separate_norm_sqrt":
            if gt_injury_risk is None or pred_injury_risk is None:
                print("⚠️  weighted_separate_norm_sqrt mode requires injury_risk scores, falling back to weighted_sqrt", flush=True)
                final_reward = self._compute_severity_reward_weighted(
                    gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                    distance_type="sqrt"
                )
            else:
                final_reward = self._compute_severity_reward_weighted_separate_norm(
                    gt_movement_score, gt_injury_risk, gt_errors,
                    pred_movement_score, pred_injury_risk, pred_errors,
                    distance_type="sqrt"
                )

        elif self.reward_mode == "two_stage":
            gt_has_errors = self._has_nontrivial_errors(gt_movement_score, gt_errors, gt_injury_risk)
            pred_has_errors = self._has_nontrivial_errors(pred_movement_score, pred_errors, pred_injury_risk)
            detection_reward = 1.0 if (gt_has_errors == pred_has_errors) else 0.0
            final_reward = (
                self.detection_weight * detection_reward +
                self.severity_weight * severity_reward
            )

        elif self.reward_mode == "two_stage_sqrt":
            gt_has_errors = self._has_nontrivial_errors(gt_movement_score, gt_errors, gt_injury_risk)
            pred_has_errors = self._has_nontrivial_errors(pred_movement_score, pred_errors, pred_injury_risk)
            detection_reward = 1.0 if (gt_has_errors == pred_has_errors) else 0.0
            sqrt_severity = self._compute_severity_reward(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="sqrt", gt_injury_risk=gt_injury_risk, pred_injury_risk=pred_injury_risk
            )
            final_reward = (
                self.detection_weight * detection_reward +
                self.severity_weight * sqrt_severity
            )

        elif self.reward_mode == "two_stage_weighted":
            gt_has_errors = self._has_nontrivial_errors(gt_movement_score, gt_errors, gt_injury_risk)
            pred_has_errors = self._has_nontrivial_errors(pred_movement_score, pred_errors, pred_injury_risk)
            detection_reward = 1.0 if (gt_has_errors == pred_has_errors) else 0.0
            weighted_severity = self._compute_severity_reward_weighted(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="linear", gt_injury_risk=gt_injury_risk, pred_injury_risk=pred_injury_risk
            )
            final_reward = (
                self.detection_weight * detection_reward +
                self.severity_weight * weighted_severity
            )

        elif self.reward_mode == "detection_correctness_severity":
            if gt_injury_risk is None or pred_injury_risk is None:
                print("⚠️  detection_correctness_severity mode requires injury_risk scores, falling back to vanilla", flush=True)
                final_reward = self._compute_severity_reward(
                    gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                    distance_type="linear"
                )
            else:
                final_reward = self._compute_severity_reward_detection_correctness_severity(
                    gt_movement_score, gt_injury_risk, gt_errors,
                    pred_movement_score, pred_injury_risk, pred_errors,
                    distance_type="linear"
                )

        elif self.reward_mode == "detection_correctness_severity_quadratic":
            if gt_injury_risk is None or pred_injury_risk is None:
                print("⚠️  detection_correctness_severity_quadratic mode requires injury_risk scores, falling back to vanilla_quadratic", flush=True)
                final_reward = self._compute_severity_reward(
                    gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                    distance_type="quadratic"
                )
            else:
                final_reward = self._compute_severity_reward_detection_correctness_severity(
                    gt_movement_score, gt_injury_risk, gt_errors,
                    pred_movement_score, pred_injury_risk, pred_errors,
                    distance_type="quadratic"
                )

        elif self.reward_mode == "detection_correctness_severity_sqrt":
            if gt_injury_risk is None or pred_injury_risk is None:
                print("⚠️  detection_correctness_severity_sqrt mode requires injury_risk scores, falling back to vanilla_sqrt", flush=True)
                final_reward = self._compute_severity_reward(
                    gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                    distance_type="sqrt"
                )
            else:
                final_reward = self._compute_severity_reward_detection_correctness_severity(
                    gt_movement_score, gt_injury_risk, gt_errors,
                    pred_movement_score, pred_injury_risk, pred_errors,
                    distance_type="sqrt"
                )

        else:
            print(f"⚠️  Unknown reward_mode: {self.reward_mode}, using vanilla", flush=True)
            final_reward = severity_reward

        # DEBUG: Print final reward for first 2 samples
        if hasattr(ThriveVLMVerifyWorker, '_debug_print_counter') and ThriveVLMVerifyWorker._debug_print_counter <= 2:
            if ThriveVLMVerifyWorker._debug_print_counter > 0:  # Only print if we printed scores above
                print(f"\nFinal Reward: {final_reward:.4f}")
                print("="*80 + "\n")

        return final_reward

    def _compute_severity_reward(
        self,
        gt_movement_score: float,
        gt_errors: dict[str, float],
        pred_movement_score: float,
        pred_errors: dict[str, float],
        distance_type: str = "linear",
        gt_injury_risk: Optional[float] = None,
        pred_injury_risk: Optional[float] = None,
    ) -> float:
        """Compute severity-based reward using distance metric.

        Args:
            distance_type: Distance function to use:
                - "linear":    |pred - gt|,        normalized by 5
                - "quadratic": (pred - gt)²,        normalized by 25 (5²)
                - "sqrt":      sqrt(|pred - gt|),   normalized by sqrt(5)

        Returns:
            float: Reward between 0.0 and 1.0 based on score accuracy
        """
        import math

        # Compute total distance
        total_distance = 0.0
        count = 0

        if distance_type == "quadratic":
            def dist(a: float, b: float) -> float:
                d = abs(a - b)
                return d * d
            max_dist = 25.0  # 5²
        elif distance_type == "sqrt":
            def dist(a: float, b: float) -> float:
                return math.sqrt(abs(a - b))
            max_dist = math.sqrt(5.0)
        else:  # linear
            def dist(a: float, b: float) -> float:
                return abs(a - b)
            max_dist = 5.0

        max_penalty = max_dist  # penalty for missing prediction

        # Movement score (Effectiveness) distance
        total_distance += dist(pred_movement_score, gt_movement_score)
        count += 1

        # Injury Risk distance (if available)
        if gt_injury_risk is not None and pred_injury_risk is not None:
            total_distance += dist(pred_injury_risk, gt_injury_risk)
            count += 1

        # Error scores distance (match by normalized name)
        normalized_gt_errors = {
            normalize_error_name(k): v for k, v in gt_errors.items()
        }
        normalized_pred_errors = {
            normalize_error_name(k): v for k, v in pred_errors.items()
        }

        # Compute distance for each ground truth error
        for norm_error_name, gt_score in normalized_gt_errors.items():
            if norm_error_name in normalized_pred_errors:
                pred_score = normalized_pred_errors[norm_error_name]
                total_distance += dist(pred_score, gt_score)
            else:
                # Missing prediction - maximum penalty
                total_distance += max_penalty
            count += 1

        # Compute average distance and convert to reward
        if count == 0:
            print(f"⚠️  No scores to compare (count=0)", flush=True)
            return 0.0

        avg_distance = total_distance / count
        reward = 1.0 - (avg_distance / max_dist)
        reward = max(0.0, reward)  # Clamp to [0, 1]

        return reward

    def _compute_severity_reward_weighted(
        self,
        gt_movement_score: float,
        gt_errors: dict[str, float],
        pred_movement_score: float,
        pred_errors: dict[str, float],
        distance_type: str = "linear",
        gt_injury_risk: Optional[float] = None,
        pred_injury_risk: Optional[float] = None,
    ) -> float:
        """Compute severity reward with per-error weighting based on GT severity.

        Errors where GT severity > 1 are weighted with nontrivial_weight.
        Errors where GT severity = 1 are weighted with trivial_weight.
        Same applies to the movement score (< 6 vs = 6).

        Uses weighted average: sum(w_i * dist_i) / sum(w_i) so reward stays in [0, 1].

        Args:
            distance_type: "linear", "quadratic", or "sqrt"

        Returns:
            float: Reward between 0.0 and 1.0
        """
        import math

        if distance_type == "quadratic":
            def dist(a: float, b: float) -> float:
                d = abs(a - b)
                return d * d
            max_dist = 25.0
        elif distance_type == "sqrt":
            def dist(a: float, b: float) -> float:
                return math.sqrt(abs(a - b))
            max_dist = math.sqrt(5.0)
        else:  # linear
            def dist(a: float, b: float) -> float:
                return abs(a - b)
            max_dist = 5.0

        total_weighted_distance = 0.0
        total_weight = 0.0

        # Movement score (Effectiveness): error_weight if non-trivial (< 3), non_error_weight if perfect (= 3)
        # Note: Effectiveness uses 1-3 scale where 3=best
        move_weight = self.error_weight if gt_movement_score < 3.0 else self.non_error_weight
        total_weighted_distance += move_weight * dist(pred_movement_score, gt_movement_score)
        total_weight += move_weight

        # Injury Risk: error_weight if non-trivial (> 1), non_error_weight if perfect (= 1)
        # Note: Injury Risk uses 1-3 scale where 1=best
        if gt_injury_risk is not None and pred_injury_risk is not None:
            injury_weight = self.error_weight if gt_injury_risk > 1.0 else self.non_error_weight
            total_weighted_distance += injury_weight * dist(pred_injury_risk, gt_injury_risk)
            total_weight += injury_weight

        # Error scores: error_weight if severity > 1, non_error_weight if severity = 1
        normalized_gt_errors = {
            normalize_error_name(k): v for k, v in gt_errors.items()
        }
        normalized_pred_errors = {
            normalize_error_name(k): v for k, v in pred_errors.items()
        }

        for norm_error_name, gt_score in normalized_gt_errors.items():
            w = self.error_weight if gt_score > 1.0 else self.non_error_weight
            if norm_error_name in normalized_pred_errors:
                pred_score = normalized_pred_errors[norm_error_name]
                total_weighted_distance += w * dist(pred_score, gt_score)
            else:
                total_weighted_distance += w * max_dist
            total_weight += w

        if total_weight == 0.0:
            return 0.0

        avg_weighted_distance = total_weighted_distance / total_weight
        reward = 1.0 - (avg_weighted_distance / max_dist)
        return max(0.0, reward)

    def _compute_severity_reward_weighted_separate_norm(
        self,
        gt_movement_score: float,  # Effectiveness: 1-3, 3=best
        gt_injury_risk: float,  # Injury Risk: 1-3, 1=best
        gt_errors: dict[str, float],  # Error severities: 1-6, 1=not present
        pred_movement_score: float,
        pred_injury_risk: float,
        pred_errors: dict[str, float],
        distance_type: str = "linear",
    ) -> float:
        """Compute severity reward with separate normalization for "no error" vs "has error" cases.

        Fixes the issue where majority "no error" samples inflate rewards by ~88%.

        Components (separately normalized):
        - no_error_penalty: Average distance for fields where GT="no error"
          - Weighted by self.non_error_weight (default: 1.0)
        - has_error_dist: Average distance for fields where GT has errors
          - Weighted by self.error_weight (default: 5.0)

        The key is that averages are computed separately for each GT class before combining.

        Score semantics:
        - Effectiveness: 1-3, where 3=best (most common, "no error")
        - Injury Risk: 1-3, where 1=best (most common, "no error")
        - Error Severities: 1-6, where 1=not present ("no error")

        Args:
            distance_type: "linear", "quadratic", or "sqrt"

        Returns:
            float: Reward between 0.0 and 1.0
        """
        import math

        # Distance function setup
        if distance_type == "quadratic":
            def dist(a: float, b: float) -> float:
                d = abs(a - b)
                return d * d
            max_dist = 25.0  # 5²
        elif distance_type == "sqrt":
            def dist(a: float, b: float) -> float:
                return math.sqrt(abs(a - b))
            max_dist = math.sqrt(5.0)
        else:  # linear
            def dist(a: float, b: float) -> float:
                return abs(a - b)
            max_dist = 5.0

        no_error_penalty_sum = 0.0
        has_error_dist_sum = 0.0
        count_gt_no_error = 0
        count_gt_error = 0

        # Effectiveness (1-3, 3=best/"no error")
        if gt_movement_score == 3.0:
            count_gt_no_error += 1
            if pred_movement_score < 3.0:
                no_error_penalty_sum += dist(pred_movement_score, 3.0)
            # else: GT=3, pred=3 → contributes nothing
        else:  # gt_movement_score < 3
            count_gt_error += 1
            has_error_dist_sum += dist(pred_movement_score, gt_movement_score)

        # Injury Risk (1-3, 1=best/"no error")
        if gt_injury_risk == 1.0:
            count_gt_no_error += 1
            if pred_injury_risk > 1.0:
                no_error_penalty_sum += dist(pred_injury_risk, 1.0)
            # else: GT=1, pred=1 → contributes nothing
        else:  # gt_injury_risk > 1
            count_gt_error += 1
            has_error_dist_sum += dist(pred_injury_risk, gt_injury_risk)

        # Error Severities (1-6, 1="not present")
        for error_name, gt_severity in gt_errors.items():
            pred_severity = pred_errors.get(error_name, 1.0)  # Default to "not present"

            if gt_severity == 1.0:
                count_gt_no_error += 1
                if pred_severity > 1.0:
                    no_error_penalty_sum += dist(pred_severity, 1.0)
                # else: GT=1, pred=1 → contributes nothing
            else:  # gt_severity > 1
                count_gt_error += 1
                has_error_dist_sum += dist(pred_severity, gt_severity)

        # Normalize each component separately by its count
        no_error_penalty = no_error_penalty_sum / count_gt_no_error if count_gt_no_error > 0 else 0.0
        has_error_dist = has_error_dist_sum / count_gt_error if count_gt_error > 0 else 0.0

        # Weighted combination using config weights
        # non_error_weight (default 1.0) for fields where GT="no error"
        # error_weight (default 5.0) for fields where GT has errors
        total_weighted_penalty = (
            self.non_error_weight * no_error_penalty +
            self.error_weight * has_error_dist
        )
        total_weight = self.non_error_weight + self.error_weight

        # Average and normalize to [0, 1]
        avg_weighted_penalty = total_weighted_penalty / total_weight
        reward = 1.0 - (avg_weighted_penalty / max_dist)

        return max(0.0, reward)

    def _compute_severity_reward_detection_correctness_severity(
        self,
        gt_movement_score: float,
        gt_injury_risk: float,
        gt_errors: dict[str, float],
        pred_movement_score: float,
        pred_injury_risk: float,
        pred_errors: dict[str, float],
        distance_type: str = "linear",
    ) -> float:
        """Compute three-component reward: detection + correctness + severity.

        Components:
        1. Detection (continuous): Per-field binary classification for error severities only
           - For each error field, check if presence/absence is correct (>1 vs =1)
           - Reward = fraction of error fields with correct presence/absence

        2. Correctness (binary): All-or-nothing exact match across ALL fields
           - 1.0 if all fields (effectiveness, injury_risk, all errors) match exactly
           - 0.0 otherwise

        3. Severity (continuous): Weighted separate normalization distance-based reward
           - Separately normalizes "GT no error" vs "GT has error" cases before combining
           - Uses error_weight and non_error_weight for weighting
           - Supports linear, quadratic, or sqrt distance

        Args:
            gt_movement_score: Ground truth effectiveness (1-3, 3=best)
            gt_injury_risk: Ground truth injury risk (1-3, 1=best)
            gt_errors: Ground truth error severities (1-6, 1=not present)
            pred_movement_score: Predicted effectiveness
            pred_injury_risk: Predicted injury risk
            pred_errors: Predicted error severities
            distance_type: "linear", "quadratic", or "sqrt"

        Returns:
            Combined reward in [0, 1]
        """
        # Component 1: Detection Reward (per-field presence/absence for error severities only)
        detection_correct = 0
        total_error_fields = 0

        for error_name in gt_errors.keys():
            gt_has_error = gt_errors[error_name] > 1.0
            pred_has_error = pred_errors.get(error_name, 1.0) > 1.0

            if gt_has_error == pred_has_error:
                detection_correct += 1
            total_error_fields += 1

        detection_reward = detection_correct / total_error_fields if total_error_fields > 0 else 0.0

        # Component 2: Correctness Reward (binary all-or-nothing)
        effectiveness_match = (pred_movement_score == gt_movement_score)
        injury_risk_match = (pred_injury_risk == gt_injury_risk)
        errors_match = all(
            pred_errors.get(name, 1.0) == gt_errors[name]
            for name in gt_errors.keys()
        )

        correctness_reward = 1.0 if (effectiveness_match and injury_risk_match and errors_match) else 0.0

        # Component 3: Severity Reward (weighted separate normalization)
        severity_reward = self._compute_severity_reward_weighted_separate_norm(
            gt_movement_score, gt_injury_risk, gt_errors,
            pred_movement_score, pred_injury_risk, pred_errors,
            distance_type=distance_type
        )

        # Combine using configured weights
        # Default: detection_weight=0.3, correctness_weight=0.2, severity_weight=0.5
        # Note: We reuse detection_weight and severity_weight from two_stage mode
        total_reward = (
            self.detection_weight * detection_reward +
            self.correctness_weight * correctness_reward +
            self.severity_weight * severity_reward
        )

        return max(0.0, min(1.0, total_reward))

    def _has_nontrivial_errors(
        self, gt_movement_score: float, gt_errors: dict[str, float], gt_injury_risk: Optional[float] = None
    ) -> bool:
        """Check if ground truth has non-trivial errors.

        Returns True if any error severity > 1 OR effectiveness score < 3 OR injury risk > 1.

        Score scales:
        - Effectiveness (movement_score): 1-3, where 3=best (no errors)
        - Injury Risk: 1-3, where 1=best (no risk)
        - Error Severities: 1-6, where 1=not present
        """
        # Effectiveness < 3 indicates errors
        if gt_movement_score < 3.0:
            return True
        # Injury risk > 1 indicates risk
        if gt_injury_risk is not None and gt_injury_risk > 1.0:
            return True
        # Any error severity > 1 indicates presence of errors
        return any(score > 1.0 for score in gt_errors.values())

    def _extract_qa_answers(self, response: str) -> list[Optional[str]]:
        """Extract Q1–Q9 answers in order from a full-exercise response.

        The expected format is repeated blocks of::

            Q: <question text>
            A: <answer>

        Returns a list of exactly 9 elements (``None`` for any missing answer).
        """
        # Match "A: <answer>" where answer is the rest of the line (single-line categorical response)
        # This prevents capturing additional text the model might generate after the answer
        pattern = re.compile(
            r'A:\s*([^\n]+)',
            re.IGNORECASE,
        )
        matches = pattern.findall(response)
        answers: list[Optional[str]] = [m.strip() for m in matches]
        # Pad or truncate to exactly 9
        while len(answers) < 9:
            answers.append(None)
        return answers[:9]

    def _compute_full_exercise_reward(
        self,
        response: str,
        ground_truth_str: str,
        distance_type: str = "linear",
    ) -> float:
        """Compute reward for the full-exercise analysis task.

        Uses ordinal distance over Q1–Q9 categorical answers plus movement score.

        Rules:
        - GT answer "unable to determine" → skip that question entirely.
        - Pred answer None / "unable to determine" / unrecognised → max penalty (score 0).
        - Movement score missing in GT → return 0.0.
        - Movement score missing in pred → score 0 for that item.
        - Final reward = mean of per-item rewards in [0, 1].

        Args:
            distance_type: ``"linear"``, ``"quadratic"``, or ``"sqrt"``.

        Returns:
            float: Reward in [0.0, 1.0].
        """
        import math

        gt_ms = self._extract_movement_score(ground_truth_str)
        if gt_ms is None:
            print(
                f"❌ Full-exercise GT missing movement_score. GT: {ground_truth_str[:200]}",
                flush=True,
            )
            return 0.0

        pred_ms = self._extract_movement_score(response)

        gt_answers = self._extract_qa_answers(ground_truth_str)
        pred_answers = self._extract_qa_answers(response)

        # Distance helpers (normalised to [0, 1])
        def _item_reward(pred_val: float, gt_val: float, max_val: float) -> float:
            d = abs(pred_val - gt_val)
            if distance_type == "quadratic":
                d = d * d
                max_d = max_val * max_val
            elif distance_type == "sqrt":
                d = math.sqrt(d)
                max_d = math.sqrt(max_val)
            else:  # linear
                max_d = max_val
            if max_d == 0.0:
                return 1.0
            return max(0.0, 1.0 - d / max_d)

        total_reward = 0.0
        count = 0

        # Movement score (range 1–6, max distance = 5)
        ms_reward = _item_reward(pred_ms, gt_ms, 5.0) if pred_ms is not None else 0.0
        total_reward += ms_reward
        count += 1

        # Q1–Q9
        for q_idx, (gt_ans, pred_ans) in enumerate(zip(gt_answers, pred_answers)):
            if gt_ans is None:
                continue
            gt_lower = gt_ans.lower().strip()
            if gt_lower == "unable to determine":
                continue  # GT is uncertain — skip

            options = Q_ORDINALS.get(q_idx, {})
            gt_ord = options.get(gt_lower)
            if gt_ord is None:
                # Unrecognised GT answer — skip to avoid punishing the model unfairly
                continue

            max_ord = float(max(options.values()))

            if pred_ans is None:
                q_reward = 0.0
            else:
                pred_lower = pred_ans.lower().strip()
                if pred_lower == "unable to determine":
                    q_reward = 0.0
                else:
                    pred_ord = options.get(pred_lower)
                    if pred_ord is None:
                        q_reward = 0.0  # Unrecognised prediction → max penalty
                    else:
                        q_reward = _item_reward(float(pred_ord), float(gt_ord), max_ord)

            total_reward += q_reward
            count += 1

        if count == 0:
            return 0.0
        return total_reward / count

    def _extract_section(self, response: str, section_name: str) -> str:
        """Extract content between [SECTION_NAME] markers.

        Args:
            response: Full model response
            section_name: Name of section (e.g., "ERRORS", "SCORES")

        Returns:
            Content of the section, or empty string if not found
        """
        # Match [SECTION_NAME]\n content \n[NEXT_SECTION or end]
        pattern = rf'\[{section_name}\]\s*\n(.*?)(?=\[|$)'
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)

        if match:
            return match.group(1).strip()
        return ""

    def _extract_movement_score(self, response: str) -> Optional[float]:
        """Extract movement score from response.

        Handles both formats:
        - New: [SCORES]\nEffectiveness: X\nInjury Risk: Y
        - Old: Movement Score: X or Movement Score: X/Y
        """
        # Try new format first: extract from [SCORES] section
        scores_section = self._extract_section(response, "SCORES")
        if scores_section:
            # Look for "Effectiveness: X"
            match = re.search(r'Effectiveness:\s*(\d+)', scores_section, re.IGNORECASE)
            if match:
                return float(match.group(1))

        # Fallback to old format: "Movement Score: X"
        match = re.search(
            r'Movement Score:\s*(\d+)(?:/\d+)?', response, re.IGNORECASE
        )
        if match:
            return float(match.group(1))

        return None

    def _extract_injury_risk(self, response: str) -> Optional[float]:
        """Extract injury risk score from response.

        Looks for "Injury Risk: X" in [SCORES] section.
        """
        scores_section = self._extract_section(response, "SCORES")
        if scores_section:
            match = re.search(r'Injury Risk:\s*(\d+)', scores_section, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_error_scores(self, response: str) -> dict[str, float]:
        """Extract error severity scores from response.

        Handles multiple formats:
        - New: [ERRORS]\nError Name: score
        - Old: "Error Severity Assessment:\n[Error Name]: score"
        - Plain: "Error Name (description): score"
        """
        errors = {}

        # Try new format first: extract from [ERRORS] section
        errors_section = self._extract_section(response, "ERRORS")

        if errors_section:
            text_to_parse = errors_section
        else:
            # Fallback to old format: "Error Severity Assessment:" section
            assessment_match = re.search(
                r'Error Severity Assessment:(.*?)(?=Movement Score:|$)',
                response,
                re.IGNORECASE | re.DOTALL
            )
            text_to_parse = assessment_match.group(1) if assessment_match else response

        # Pattern to match error lines with optional brackets, descriptions, and formatting
        # Matches: "[Error Name]: 5", "Error Name (description): 5", "Error Name: 5", "-Error Name: 5"
        pattern = r'^[-•*\s]*\[?([^\]:(]+?)(?:\([^\)]*\))?\]?\s*:\s*(\d+)\s*$'

        for line in text_to_parse.strip().split('\n'):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip quoted feedback lines (they contain colons but aren't scores)
            if line.startswith('"') or line.startswith('"'):
                continue

            match = re.match(pattern, line)
            if match:
                # Extract error name and strip any trailing whitespace
                error_name = match.group(1).strip()
                score = float(match.group(2))
                errors[error_name] = score

        return errors


class ThriveVLMJudgeWorker:
    """Worker that uses an LLM judge (via vLLM) to score Thrive VLM responses.

    This class creates a VllmGeneration instance using the provided cluster.
    It's designed to be created inside a Ray actor environment where the cluster
    is created during the actor's __init__.
    """

    def __init__(
        self,
        judge_config: JudgeConfig,
        cluster: Optional[RayVirtualCluster],
    ) -> None:
        """Initialize the judge worker and create vLLM generation instance.

        Args:
            judge_config: Configuration for the judge model
            cluster: Virtual cluster for GPU allocation (None if colocated - not yet supported)
        """
        import os

        from nemo_rl.prompts.thrive_judge_rubrics import (
            format_judge_prompt,
            get_rubric,
            parse_judge_output,
        )

        logging.getLogger("thrive_vlm_judge_worker").setLevel(logging.CRITICAL)

        self.judge_config = judge_config
        self.rubric_type = judge_config["rubric_type"]
        self.custom_rubric_path = judge_config.get("custom_rubric_path", None)
        self.output_format = judge_config.get("output_format", "json")
        self.include_ground_truth = judge_config.get("include_ground_truth", True)
        self.batch_size = judge_config.get("batch_size", 4)

        # Load rubric template
        self.rubric = get_rubric(self.rubric_type, self.custom_rubric_path)

        # Store parse function
        self.parse_judge_output = parse_judge_output
        self.format_judge_prompt = format_judge_prompt

        # Create vLLM generation for judge
        if cluster is None:
            raise NotImplementedError(
                "Judge colocation mode is not yet supported in self-contained environment architecture. "
                "Please set judge.colocated.enabled: false and provide judge.resources config."
            )

        print(f"🔨 Initializing judge vLLM with model: {judge_config['model_name']}")

        # Load tokenizer for judge model to configure generation properly
        from nemo_rl.algorithms.utils import get_tokenizer
        from nemo_rl.models.generation import configure_generation_config

        judge_tokenizer = get_tokenizer(
            judge_config["generation"].get("tokenizer", {"name": judge_config["model_name"]}),
            get_processor=False,
        )

        # Build vLLM config from judge config
        vllm_config = {
            "backend": "vllm",
            "model_name": judge_config["model_name"],
            "vllm_cfg": judge_config["generation"]["vllm_cfg"],
            "max_new_tokens": judge_config["generation"].get("max_new_tokens", 2048),
            "temperature": judge_config["generation"].get("temperature", 0.0),
            "top_p": judge_config["generation"].get("top_p", 1.0),
            "top_k": judge_config["generation"].get("top_k", None),
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": judge_config["colocated"],
        }

        # Add tokenizer config if present
        if "tokenizer" in judge_config["generation"]:
            vllm_config["tokenizer"] = judge_config["generation"]["tokenizer"]

        # Add vllm_kwargs if present (from generation config)
        if "vllm_kwargs" in judge_config["generation"]:
            vllm_config["vllm_kwargs"] = judge_config["generation"]["vllm_kwargs"]

        # Configure generation config to set internal fields like _pad_token_id
        vllm_config = configure_generation_config(vllm_config, judge_tokenizer, is_eval=True)

        # Remove CUDA_VISIBLE_DEVICES to let ray control GPU allocation
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self.judge_vllm = VllmGeneration(
            cluster=cluster,
            config=vllm_config,
            name_prefix="thrive_judge",
        )

        print("✅ Judge vLLM initialized successfully")

    def judge_batch(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        task_types: Optional[list[str]] = None,
        video_contexts: Optional[list[str]] = None,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """Judge a batch of responses using the LLM judge.

        Args:
            pred_responses: Predicted responses from the model
            ground_truths: Ground truth responses
            task_types: Task type for each response ("repetition" or "full_exercise")
            video_contexts: Optional video context descriptions

        Returns:
            Tuple of (rewards, parsed_details) where:
                - rewards: List of total reward scores (floats between 0 and 1)
                - parsed_details: List of dicts with detailed scores and reasoning
        """
        if task_types is None:
            task_types = ["repetition"] * len(pred_responses)
        if video_contexts is None:
            video_contexts = ["Video of exercise performance"] * len(pred_responses)

        # Format judge prompts
        judge_prompts = []
        for pred, gt, task_type, context in zip(
            pred_responses, ground_truths, task_types, video_contexts
        ):
            prompt = self.format_judge_prompt(
                rubric=self.rubric,
                video_context=context,
                model_response=pred,
                ground_truth=gt if self.include_ground_truth else None,
                task_type=task_type,
            )
            judge_prompts.append(prompt)

        # Call judge vLLM to get scores
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        # Create input data for vLLM
        judge_data = BatchedDataDict(prompts=judge_prompts)

        # DEBUG: Print judge input prompts for first 2 samples every step
        print("\n" + "="*80)
        print("JUDGE INPUT PROMPTS (first 2 samples):")
        for i, prompt in enumerate(judge_prompts[:2]):
            print(f"\n[Sample {i}] Judge Prompt:")
            print(prompt[:1000] + ("..." if len(prompt) > 1000 else ""))
            print("-" * 80)

        # Generate judge evaluations (greedy decoding for consistency)
        judge_outputs = self.judge_vllm.generate_text(judge_data, greedy=True)

        # DEBUG: Print judge outputs for first 2 samples every step
        print("\n" + "="*80)
        print("JUDGE OUTPUTS (first 2 samples):")
        for i, output in enumerate(judge_outputs["texts"][:2]):
            print(f"\n[Sample {i}] Judge Output:")
            print(output)
            print("-" * 80)

        # Parse judge outputs to extract scores
        rewards = []
        parsed_details = []
        for i, output_text in enumerate(judge_outputs["texts"]):
            try:
                parsed = self.parse_judge_output(output_text)
                reward = parsed["total_reward"]
                parsed_details.append(parsed)

            except Exception as e:
                print(f"⚠️  Failed to parse judge output for sample {i}: {e}")
                print(f"Judge output: {output_text[:200]}")
                reward = 0.5  # Fallback to neutral score
                parsed_details.append({"total_reward": reward})
            rewards.append(reward)

        return rewards, parsed_details

    def verify(
        self,
        pred_responses: list[str],
        ground_truths: list[str],
        task_types: Optional[list[str]] = None,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        """Verify responses using the judge (alias for judge_batch for compatibility).

        This method maintains the same interface as ThriveVLMVerifyWorker.

        Args:
            pred_responses: Predicted responses from the model
            ground_truths: Ground truth responses
            task_types: Task type for each response ("repetition" or "full_exercise")

        Returns:
            Tuple of (rewards, parsed_details)
        """
        return self.judge_batch(pred_responses, ground_truths, task_types)


class ThriveVLMEnvironmentMetadata(TypedDict):
    ground_truth: str
    task_type: str  # "repetition" or "full_exercise"


@ray.remote(max_restarts=-1, max_task_retries=-1)
class ThriveVLMEnvironment(EnvironmentInterface):
    def __init__(
        self,
        cfg: ThriveVLMEnvConfig,
    ):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]

        # Always create rule-based verify workers
        self.verify_workers = [
            ThriveVLMVerifyWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(cfg)
            for _ in range(self.num_workers)
        ]

        # Initialize judge if enabled (self-contained architecture)
        self.use_judge = cfg.get("judge", {}).get("enabled", False) if "judge" in cfg else False
        self.judge_worker = None
        self.judge_weight = 0.5  # Default

        if self.use_judge:
            judge_config = cfg["judge"]
            print(f"🔨 Initializing ThriveVLM judge (self-contained)...")

            # Create judge virtual cluster (if not colocated)
            if not judge_config["colocated"]["enabled"]:
                judge_resources = judge_config.get("resources", {"gpus_per_node": 2, "num_nodes": 1})
                self.judge_virtual_cluster = RayVirtualCluster(
                    name="thrive_vlm_judge_cluster",
                    bundle_ct_per_node_list=[judge_resources["gpus_per_node"]] * judge_resources["num_nodes"],
                    use_gpus=True,
                    num_gpus_per_node=judge_resources["gpus_per_node"],
                    max_colocated_worker_groups=1,
                )
                print(f"  ✓ Created dedicated judge cluster: {judge_resources['num_nodes']} nodes × {judge_resources['gpus_per_node']} GPUs")
            else:
                # For colocated mode, the cluster will be shared with generation/policy
                # The cluster must be created externally and passed via colocation mechanism
                self.judge_virtual_cluster = None
                print(f"  ⚠ Judge colocation enabled - cluster management not supported in self-contained mode")
                print(f"     Colocation target: {judge_config['colocated'].get('colocation_target', 'generation')}")

            # Create judge worker (which will create VllmGeneration internally)
            self.judge_worker = ThriveVLMJudgeWorker(judge_config, self.judge_virtual_cluster)
            self.judge_weight = judge_config.get("judge_weight", 0.5)
            print(f"  ✓ Judge initialized with weight={self.judge_weight}")
        else:
            self.judge_virtual_cluster = None

    def shutdown(self) -> None:
        # shutdown verify workers
        for worker in self.verify_workers:
            ray.kill(worker)

        # shutdown judge worker if it exists
        # Note: judge_worker is not a Ray actor, so we don't need to kill it
        # The VllmGeneration workers inside it will be cleaned up when the cluster is destroyed

    def step(  # type: ignore[override]
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[ThriveVLMEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the thrive-vlm environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs.
            metadata: list[ThriveVLMEnvironmentMetadata]. Ground truth scores in JSON format.

        Returns:
            EnvironmentReturn: A tuple containing observations, metadata, stop strings, rewards, and done flags.
        """
        # Extract the assistant's responses from the message history
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            full_response = "".join(assistant_responses)

            # Strip reasoning trace for reasoning models
            # Remove everything from start up to and including </think> tag
            cleaned_response = re.sub(r'^.*?</think>\s*', '', full_response, flags=re.DOTALL)

            assistant_response_batch.append(cleaned_response.strip())

        ground_truths = [g["ground_truth"] for g in metadata]
        task_types = [g.get("task_type", "repetition") for g in metadata]

        # DEBUG: Print generation outputs for first 2 samples every step
        print("\n" + "="*80)
        print("GENERATION OUTPUTS (first 2 samples):")
        for i, response in enumerate(assistant_response_batch[:2]):
            print(f"\n[Sample {i}] Generated Response:")
            print(response[:500] + ("..." if len(response) > 500 else ""))
            print("-" * 80)

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)
        chunked_task_types = chunk_list_to_workers(task_types, self.num_workers)

        # Process each chunk in parallel with verify workers
        verify_futures = [
            self.verify_workers[i].verify.remote(chunk, ground_truth_chunk, task_type_chunk)
            for i, (chunk, ground_truth_chunk, task_type_chunk) in enumerate(
                zip(
                    chunked_assistant_response_batch,
                    chunked_ground_truths,
                    chunked_task_types,
                )
            )
        ]

        verify_results = ray.get(verify_futures)
        # flatten the verify results
        verify_results = [item for sublist in verify_results for item in sublist]

        # If judge is enabled, also get judge rewards
        if self.use_judge and self.judge_worker:
            # Call the single judge worker with the full batch
            # (vLLM handles parallelism internally via DP)
            # Note: judge_worker is not a Ray actor, so we call it directly
            judge_results, judge_details = self.judge_worker.verify(
                assistant_response_batch, ground_truths, task_types
            )

            # Combine verify and judge rewards using judge_weight
            results = [
                (1 - self.judge_weight) * v_reward + self.judge_weight * j_reward
                for v_reward, j_reward in zip(verify_results, judge_results)
            ]

            # DEBUG: Print reward combination for first 2 samples every step
            for idx in range(min(2, len(results))):
                details = judge_details[idx]

                print("\n" + "="*80)
                print(f"REWARD COMBINATION [Sample {idx}]:")
                print(f"\n  Rule-Based Verify Reward: {verify_results[idx]:.4f}")
                print(f"\n  Judge Scores:")
                # Print individual judge scores if available
                if "tone" in details:
                    print(f"    - Tone:               {details['tone']:.4f}")
                if "informativeness" in details:
                    print(f"    - Informativeness:    {details['informativeness']:.4f}")
                if "feedback_agreement" in details:
                    print(f"    - Feedback Agreement: {details['feedback_agreement']:.4f}")
                if "analysis_agreement" in details:
                    print(f"    - Analysis Agreement: {details['analysis_agreement']:.4f}")
                print(f"    - Total Judge Reward: {judge_results[idx]:.4f}")

                print(f"\n  Combination:")
                print(f"    - Judge Weight: {self.judge_weight:.2f}")
                print(f"    - Formula: (1 - {self.judge_weight:.2f}) × {verify_results[idx]:.4f} + {self.judge_weight:.2f} × {judge_results[idx]:.4f}")
                print(f"    - Final Reward: {results[idx]:.4f}")
                print("="*80 + "\n")
        else:
            # Use only verify rewards
            results = verify_results

        observations = [
            {
                "role": "environment",
                "content": f"Environment: reward={result:.3f}",
            }
            for result in results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor(results).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Computes metrics for this environment given a global rollout batch."""
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences

        # Compute average reward for correctly ended sequences
        if (batch["is_end"] == 1).float().sum() > 0:
            avg_reward_correct = (
                batch["rewards"][batch["is_end"] == 1].float().mean().item()
            )
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["is_end"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            avg_reward_correct = 0.0
            correct_solution_generation_lengths = 0

        metrics = {
            "avg_reward": batch["rewards"].mean().item(),
            "avg_reward_correct_endings": avg_reward_correct,
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
