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
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers


class ThriveVLMEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env
    reward_mode: Optional[str]  # "vanilla", "vanilla_quadratic", "vanilla_sqrt", "weighted", "weighted_quadratic", "weighted_sqrt", "two_stage", "two_stage_sqrt", "two_stage_weighted" (default: "vanilla")
    # Weighted mode parameters
    error_weight: Optional[float]  # Weight for errors with GT severity > 1 or movement < 6 (default: 1.0)
    non_error_weight: Optional[float]  # Weight for errors with GT severity = 1 or movement = 6 (default: 0.5)
    # Two-stage mode parameters
    detection_weight: Optional[float]  # Weight for detection accuracy (default: 0.3)
    severity_weight: Optional[float]  # Weight for severity accuracy (default: 0.7)
    # Full-exercise analysis reward mode ("vanilla", "vanilla_quadratic", "vanilla_sqrt"; default: "vanilla")
    full_exercise_reward_mode: Optional[str]


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
        for response, ground_truth_str, task_type in zip(pred_responses, ground_truths, task_types):
            try:
                reward = self._compute_reward(response, ground_truth_str, task_type=task_type)
                results.append(float(reward))
            except Exception:
                results.append(0.0)
        return results

    def _compute_reward(
        self, response: str, ground_truth_str: str, task_type: str = "repetition"
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
        gt_errors = self._extract_error_scores(ground_truth_str)

        if gt_movement_score is None:
            print(f"❌ Ground truth missing movement_score. Ground truth text: {ground_truth_str[:200]}", flush=True)
            return 0.0

        # Parse predicted scores from response
        pred_movement_score = self._extract_movement_score(response)
        pred_errors = self._extract_error_scores(response)

        if pred_movement_score is None:
            print(f"❌ Failed to extract movement score from response: {response[:300]}", flush=True)
            return 0.0

        # Compute base severity reward (distance-based)
        severity_reward = self._compute_severity_reward(
            gt_movement_score, gt_errors, pred_movement_score, pred_errors
        )

        # Apply reward mode
        if self.reward_mode == "vanilla":
            final_reward = severity_reward

        elif self.reward_mode == "vanilla_quadratic":
            final_reward = self._compute_severity_reward(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="quadratic"
            )

        elif self.reward_mode == "vanilla_sqrt":
            final_reward = self._compute_severity_reward(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="sqrt"
            )

        elif self.reward_mode == "weighted":
            final_reward = self._compute_severity_reward_weighted(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="linear"
            )

        elif self.reward_mode == "weighted_quadratic":
            final_reward = self._compute_severity_reward_weighted(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="quadratic"
            )

        elif self.reward_mode == "weighted_sqrt":
            final_reward = self._compute_severity_reward_weighted(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="sqrt"
            )

        elif self.reward_mode == "two_stage":
            gt_has_errors = self._has_nontrivial_errors(gt_movement_score, gt_errors)
            pred_has_errors = self._has_nontrivial_errors(pred_movement_score, pred_errors)
            detection_reward = 1.0 if (gt_has_errors == pred_has_errors) else 0.0
            final_reward = (
                self.detection_weight * detection_reward +
                self.severity_weight * severity_reward
            )

        elif self.reward_mode == "two_stage_sqrt":
            gt_has_errors = self._has_nontrivial_errors(gt_movement_score, gt_errors)
            pred_has_errors = self._has_nontrivial_errors(pred_movement_score, pred_errors)
            detection_reward = 1.0 if (gt_has_errors == pred_has_errors) else 0.0
            sqrt_severity = self._compute_severity_reward(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="sqrt"
            )
            final_reward = (
                self.detection_weight * detection_reward +
                self.severity_weight * sqrt_severity
            )

        elif self.reward_mode == "two_stage_weighted":
            gt_has_errors = self._has_nontrivial_errors(gt_movement_score, gt_errors)
            pred_has_errors = self._has_nontrivial_errors(pred_movement_score, pred_errors)
            detection_reward = 1.0 if (gt_has_errors == pred_has_errors) else 0.0
            weighted_severity = self._compute_severity_reward_weighted(
                gt_movement_score, gt_errors, pred_movement_score, pred_errors,
                distance_type="linear"
            )
            final_reward = (
                self.detection_weight * detection_reward +
                self.severity_weight * weighted_severity
            )

        else:
            print(f"⚠️  Unknown reward_mode: {self.reward_mode}, using vanilla", flush=True)
            final_reward = severity_reward

        return final_reward

    def _compute_severity_reward(
        self,
        gt_movement_score: float,
        gt_errors: dict[str, float],
        pred_movement_score: float,
        pred_errors: dict[str, float],
        distance_type: str = "linear",
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

        # Movement score distance
        total_distance += dist(pred_movement_score, gt_movement_score)
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

        # Movement score: error_weight if non-trivial (< 6), non_error_weight if perfect (= 6)
        move_weight = self.error_weight if gt_movement_score < 6.0 else self.non_error_weight
        total_weighted_distance += move_weight * dist(pred_movement_score, gt_movement_score)
        total_weight += move_weight

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

    def _has_nontrivial_errors(
        self, gt_movement_score: float, gt_errors: dict[str, float]
    ) -> bool:
        """Check if ground truth has non-trivial errors.

        Returns True if any error severity > 1 OR movement score < 6.
        """
        if gt_movement_score < 6.0:
            return True
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

    def _extract_movement_score(self, response: str) -> Optional[float]:
        """Extract movement score from response.

        Handles formats like:
        - Movement Score: 6
        - Movement Score: 6/6
        """
        match = re.search(
            r'Movement Score:\s*(\d+)(?:/\d+)?', response, re.IGNORECASE
        )
        if match:
            return float(match.group(1))
        return None

    def _extract_error_scores(self, response: str) -> dict[str, float]:
        """Extract error severity scores from response.

        Handles multiple formats:
        - With section header: "Error Severity Assessment:\n[Error Name]: score"
        - Without header: "Error Name (description): score"
        - With brackets: "[Error Name]: score"
        - Plain format: "Error Name: score"
        """
        errors = {}

        # Try to find "Error Severity Assessment:" section first (for model responses)
        assessment_match = re.search(
            r'Error Severity Assessment:(.*?)(?=Movement Score:|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )

        # Use the section if found, otherwise parse entire text (for ground truth)
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


class ThriveVLMEnvironmentMetadata(TypedDict):
    ground_truth: str
    task_type: str  # "repetition" or "full_exercise"


@ray.remote(max_restarts=-1, max_task_retries=-1)
class ThriveVLMEnvironment(EnvironmentInterface):
    def __init__(self, cfg: ThriveVLMEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        self.workers = [
            ThriveVLMVerifyWorker.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(cfg)
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

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

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_ground_truths = chunk_list_to_workers(ground_truths, self.num_workers)
        chunked_task_types = chunk_list_to_workers(task_types, self.num_workers)

        # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, ground_truth_chunk, task_type_chunk)
            for i, (chunk, ground_truth_chunk, task_type_chunk) in enumerate(
                zip(
                    chunked_assistant_response_batch,
                    chunked_ground_truths,
                    chunked_task_types,
                )
            )
        ]

        results = ray.get(futures)

        # flatten the results
        results = [item for sublist in results for item in sublist]
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
