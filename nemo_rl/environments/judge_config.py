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

from typing import Any, Literal, NotRequired, TypedDict

from nemo_rl.models.generation.interfaces import OptionalResourcesConfig


class JudgeColocationConfig(TypedDict):
    """Configuration for colocating judge with policy or generation."""

    enabled: bool
    # Target for colocation: "policy" or "generation"
    # If "policy", judge shares GPUs with training cluster
    # If "generation", judge shares GPUs with vLLM generation cluster
    colocation_target: NotRequired[Literal["policy", "generation"]]


class JudgeVllmConfig(TypedDict):
    """vLLM-specific configuration for judge inference."""

    tensor_parallel_size: int
    pipeline_parallel_size: NotRequired[int]
    expert_parallel_size: NotRequired[int]
    gpu_memory_utilization: float
    max_model_len: int
    async_engine: bool
    precision: NotRequired[str]
    kv_cache_dtype: NotRequired[Literal["auto", "fp8", "fp8_e4m3"]]
    enforce_eager: NotRequired[bool]


class JudgeGenerationConfig(TypedDict):
    """Generation configuration for judge."""

    backend: Literal["vllm"]  # Currently only vLLM is supported
    vllm_cfg: JudgeVllmConfig
    max_new_tokens: NotRequired[int]  # Default: 2048
    temperature: NotRequired[float]  # Default: 0.0 (greedy)
    top_p: NotRequired[float]  # Default: 1.0
    top_k: NotRequired[int | None]  # Default: None


class JudgeConfig(TypedDict):
    """Configuration for LLM-as-a-judge."""

    enabled: bool
    # Model to use for judging
    model_name: str
    # Rubric type: predefined rubrics or "custom" to load from file
    # Predefined options: "repetition_analysis", "movement_assessment", "error_detection", "two_stage", "full_exercise"
    rubric_type: str
    # Path to custom rubric file (required if rubric_type="custom")
    custom_rubric_path: NotRequired[str | None]
    # Generation configuration for judge
    generation: JudgeGenerationConfig
    # Colocation settings
    colocated: JudgeColocationConfig
    # Resources when not colocated
    resources: NotRequired[OptionalResourcesConfig]
    # Output format for judge scores
    # "json" expects structured JSON output with scores
    # "text" parses scores from natural language
    output_format: NotRequired[Literal["json", "text"]]
    # Whether to include ground truth in judge prompt for calibration
    include_ground_truth: NotRequired[bool]
    # Batch size for judge inference
    batch_size: NotRequired[int]
    # Weight for combining judge rewards with verify rewards (0-1)
    # judge_weight=0.5 means final_reward = 0.5*judge_reward + 0.5*verify_reward
    judge_weight: NotRequired[float]
    # Additional vLLM kwargs
    vllm_kwargs: NotRequired[dict[str, Any]]
