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

import json
import logging
from typing import Any, Optional

from datasets import load_from_disk
from PIL import Image, ImageOps
from transformers.video_utils import VideoMetadata

logger = logging.getLogger(__name__)

from nemo_rl.data import ResponseDatasetConfig
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import PROCESSOR_REGISTRY


def format_thrive_vlm_grpo_dataset(
    example: dict[str, Any], return_pil: bool = False
) -> dict[str, Any]:
    """Format the THRIVE-VLM video dataset for GRPO training.

    This is designed specifically for GRPO where we format the dataset
    with user questions and assistant answers for RL training.

    Args:
        example: Dataset example containing video and text data
        return_pil: If True, return raw video frames (list of PIL Images)
                   If False, use video path/URL directly

    Returns:
        Formatted message log dictionary with user question and assistant answer
    """
    # Determine if this is a comparison sample (two videos) or single-video sample
    is_comparison = bool(example.get("video_frames_a")) and bool(example.get("video_frames_b"))

    # Extract fps (shared across all video types)
    if "fps" in example or "sample_fps" in example:
        fps_value = example.get("fps", example.get("sample_fps", 10.0))
        if isinstance(fps_value, str):
            fps_value = float(fps_value)
    else:
        logger.warning("No fps found in example, defaulting to 10.0")
        fps_value = 10.0

    def _load_video(frame_paths, apply_flip=False):
        """Load video frames and optionally flip. Returns (frames, VideoMetadata content dict)."""
        frames = frame_paths
        if isinstance(frames, (list, tuple)) and len(frames) > 0 and isinstance(frames[0], str) and return_pil:
            frames = [Image.open(p).convert("RGB") for p in frames]
            if apply_flip:
                frames = [ImageOps.mirror(f) for f in frames]

        content = {"type": "video", "video": frames}

        if isinstance(frames, (list, tuple)) and len(frames) > 0:
            first_frame = frames[0]
            if isinstance(first_frame, str):
                temp_frame = Image.open(first_frame).convert("RGB")
                w, h = temp_frame.size
            elif hasattr(first_frame, "size"):
                w, h = first_frame.size
            else:
                h, w = first_frame.shape[-2:]
            content["video_metadata"] = VideoMetadata(
                total_num_frames=len(frames), fps=fps_value,
                width=w, height=h, frames_indices=list(range(len(frames))),
            )

        if "max_pixels" in example:
            content["max_pixels"] = int(example["max_pixels"])
        if "min_pixels" in example:
            content["min_pixels"] = int(example["min_pixels"])

        return frames, content

    need_flip = example.get("need_to_flip", False)

    # Blocklist: comparison samples that cause vLLM multi-video scheduling deadlocks
    _COMPARISON_BLOCKLIST = {
        "10023_60884082",
        "12000_61267072",
        "10015_57393662",
        "15009_57983650|57983629",
    }

    # Check blocklist — blocked comparison samples become single-video with zero loss
    # NOTE: Commented out to test if vLLM encoder_budget.py fix resolves the hang
    _is_blocklisted = False
    # if is_comparison:
    #     _sid = example.get("session_id", "")
    #     _eid = example.get("exercise_id", "")
    #     _sample_key = f"{_eid}_{_sid}"
    #     if _sample_key in _COMPARISON_BLOCKLIST:
    #         _is_blocklisted = True
    #         is_comparison = False
    #         example["video_frames"] = example["video_frames_a"]
    #         example["dataset_type"] = "repetition_severity"

    if is_comparison:
        # # Cap frames per video in comparison samples to avoid vLLM vision encoder OOM
        # MAX_FRAMES_PER_COMPARISON_VIDEO = 64
        #
        # def _subsample(frame_paths, max_frames):
        #     if len(frame_paths) <= max_frames:
        #         return frame_paths
        #     indices = [int(i * (len(frame_paths) - 1) / (max_frames - 1)) for i in range(max_frames)]
        #     return [frame_paths[j] for j in indices]
        #
        # video_value_a = _subsample(example["video_frames_a"], MAX_FRAMES_PER_COMPARISON_VIDEO)
        # video_value_b = _subsample(example["video_frames_b"], MAX_FRAMES_PER_COMPARISON_VIDEO)
        video_value_a = example["video_frames_a"]
        video_value_b = example["video_frames_b"]
        _, video_content_a = _load_video(video_value_a, apply_flip=need_flip)
        _, video_content_b = _load_video(video_value_b, apply_flip=need_flip)
        video_contents = [video_content_a, video_content_b]
        # Use video_frames_a for sample_id extraction fallback
        video_value = video_value_a
    else:
        # Handle video, image, or text-only samples
        video_value = example.get("video") or example.get("video_frames")

        if video_value:
            video_value, video_content = _load_video(video_value, apply_flip=need_flip)
            video_contents = [video_content]
        else:
            # No video — check for image content
            image_value = example.get("image") or example.get("images")
            video_contents = []
            video_value = None
            if image_value:
                if not isinstance(image_value, (list, tuple)):
                    image_value = [image_value]
                if len(image_value) > 0 and isinstance(image_value[0], str) and return_pil:
                    image_value = [Image.open(p).convert("RGB") for p in image_value]
                    if need_flip:
                        image_value = [ImageOps.mirror(f) for f in image_value]
                for img in image_value:
                    video_contents.append({"type": "image", "image": img})
            # Text-only samples: video_contents stays empty

    # Extract question text and answer from messages structure or direct fields
    # Dataset format: messages[0] = system prompt, messages[1] = user question, messages[2] = assistant answer (GT)
    system_prompt = ""
    question_text = ""
    assistant_content = ""

    if "messages" in example:
        messages = example["messages"]
        if len(messages) >= 1:
            # First message: system prompt
            if messages[0].get("role") == "system":
                system_prompt = messages[0].get("content", "")
        if len(messages) >= 2:
            # Second message: user question
            if messages[1].get("role") == "user":
                content = messages[1].get("content", "")
                if isinstance(content, str):
                    question_text = content
                elif isinstance(content, list):
                    # Extract text from list of content items
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            question_text = item.get("text", "")
                            break
        if len(messages) >= 3:
            # Third message: assistant answer (ground truth)
            if messages[2].get("role") == "assistant":
                assistant_content = str(messages[2].get("content", "")).strip()

    # Fallback to direct fields if messages not found
    if not question_text:
        question_text = example.get("question", example.get("problem", ""))
    if not assistant_content:
        assistant_content = str(example.get("answer", example.get("solution", ""))).strip()

    # For GRPO: construct messages with system prompt (if present), user question with video, and GT answer
    result_messages = []

    # Add system prompt if it exists
    if system_prompt:
        result_messages.append({"role": "system", "content": system_prompt})

    # Add user question with video(s)
    user_content = video_contents + [
        {
            "type": "text",
            "text": str(question_text),
        },
    ]
    result_messages.append({"role": "user", "content": user_content})

    # Add assistant ground truth
    result_messages.append({
        "role": "assistant",
        "content": assistant_content,
    })

    # Detect task type per-sample so mixed datasets work transparently.
    # Primary: check dataset_type field (e.g. "full_exercise", "repetition_severity").
    # Aux types pass through as-is. Fallback: check ratings metadata for Q1-Q9 keys.
    _AUX_DATASET_TYPES = {
        "video_mcqa", "image_mcqa", "phase_sequencing_mcqa",
        "muscle_exercise_mcqa", "error_correction_mcqa", "error_recognition",
        "exercise_name_identification", "keypoint_prediction", "keypoint_labeling",
    }
    dataset_type = example.get("dataset_type", "")
    if dataset_type == "comparison":
        task_type = "comparison"
    elif dataset_type == "full_exercise":
        task_type = "full_exercise"
    elif dataset_type in _AUX_DATASET_TYPES:
        task_type = dataset_type
    elif isinstance(example.get("ratings"), dict) and "q1_consistency" in example["ratings"]:
        task_type = "full_exercise"
    else:
        task_type = "repetition"

    # Extract sample identifier from frame paths or dataset fields
    sample_id = ""
    if "session_id" in example:
        sample_id = f"{example.get('exercise_id', '')}_{example.get('session_id', '')}"
    elif isinstance(video_value, (list, tuple)) and len(video_value) > 0 and isinstance(video_value[0], str):
        # Extract folder name from frame path
        import os
        sample_id = os.path.basename(os.path.dirname(os.path.dirname(video_value[0])))

    extra_env_info = {
        "ground_truth": assistant_content,
        "task_type": task_type,
        "sample_id": sample_id,
    }
    if task_type == "comparison":
        extra_env_info["expected_verdict"] = example.get("expected_verdict", "")

    ret = {
        "messages": result_messages,
        "task_name": "thrive-vlm",
        "task_type": task_type,
        "extra_env_info": extra_env_info,
    }

    if _is_blocklisted:
        ret["_skip"] = True

    return ret


def prepare_thrive_vlm_grpo_dataset(
    split: str = "train",
    dataset_name: str = "thrive-vlm",
    task_name: Optional[str] = None,
):
    """Prepare THRIVE-VLM dataset for GRPO training.

    Args:
        split: Dataset split to load (train, validation, test)
        dataset_name: Local path to dataset directory (saved with save_to_disk)
        task_name: Optional task name override

    Returns:
        Dictionary with train and validation splits
    """
    if task_name is None:
        task_name = "thrive-vlm"

    # Load dataset from disk using load_from_disk
    raw = load_from_disk(dataset_name)

    # Check if raw is a DatasetDict or a single Dataset
    if hasattr(raw, "keys") and callable(raw.keys):
        # It's a DatasetDict with splits
        if split == "train":
            train_dataset = raw["train"]
            val_dataset = raw.get("validation", raw.get("val", raw["train"]))
        else:
            if split not in raw:
                raise ValueError(
                    f"Split '{split}' not found. Available: {list(raw.keys())}"
                )

            train_dataset = raw[split]
            val_dataset = raw[split]
    else:
        # It's a single Dataset, use it for both train and validation
        train_dataset = raw
        val_dataset = raw

    # Format - add task_name column if not present
    # Note: Keep dataset_type column as is (it contains data categories like "repetition", "severity")
    # and add a separate task_name column for GRPO environment matching
    if "task_name" not in train_dataset.column_names:
        train_dataset = train_dataset.add_column("task_name", [task_name] * len(train_dataset))

    if "task_name" not in val_dataset.column_names:
        val_dataset = val_dataset.add_column("task_name", [task_name] * len(val_dataset))

    return {
        "train": train_dataset,
        "validation": val_dataset,
    }


class ThriveVLMGRPODataset:
    """Dataset class for THRIVE-VLM video question answering for GRPO training.

    This dataset handles video inputs with question-answer pairs for VLM RL training.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        dataset_path: Optional[str] = None,
        **kwargs,  # Accept other params from config
    ):
        if split not in ["train", "validation"]:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train' or 'validation'."
            )
        self.task_name = "thrive-vlm"

        # Use dataset_path if provided, otherwise use dataset_name as the path
        path = dataset_path if dataset_path is not None else dataset_name

        self.formatted_ds = prepare_thrive_vlm_grpo_dataset(
            split=split,
            task_name=self.task_name,
            dataset_name=path,
        )

        self.task_spec = TaskDataSpec(
            task_name="thrive-vlm",
        )

        # Initialize these to None, will be set by set_task_spec and set_processor
        self.data_config = None
        self.processor = None
        self.val_dataset = self.formatted_ds.get("validation")

    @property
    def dataset(self):
        """Expose the training dataset."""
        return self.formatted_ds["train"]

    def set_task_spec(
        self, data_config: ResponseDatasetConfig
    ):
        """Set task specification from data config.

        Args:
            data_config: Configuration dict containing system_prompt_file and prompt_file
        """
        self.data_config = data_config
        system_prompt_file = self.data_config.get("system_prompt_file", None)
        prompt_file = self.data_config.get("prompt_file", None)
        self.task_spec = TaskDataSpec(
            task_name=self.task_name,
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )

    def set_processor(self):
        """Set the data processor to grpo_processor.

        For Thrive VLM GRPO, we use grpo_processor which:
        - Filters assistant messages from vllm_content (only system+user for generation)
        - Keeps full conversation in message_log (for logprobs after generation)
        - Uses format_thrive_vlm_grpo_dataset as datum_preprocessor
        """
        from nemo_rl.data.processors import grpo_processor

        self.processor = grpo_processor
