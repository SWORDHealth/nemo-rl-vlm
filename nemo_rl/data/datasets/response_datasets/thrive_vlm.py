

from typing import Any, Optional

from datasets import load_from_disk
from PIL import Image, ImageOps
from transformers.video_utils import VideoMetadata

from nemo_rl.data import ResponseDatasetConfig
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import PROCESSOR_REGISTRY


def format_thrive_vlm_dataset(
    example: dict[str, Any], return_pil: bool = False
) -> dict[str, Any]:
    """Format the THRIVE-VLM dataset into an OpenAI-API-like message log.

    Supports datasets with different modalities:
    - Video samples (with 'video' or 'video_frames' fields)
    - Image samples (with 'image' or 'images' fields)
    - Text-only samples (no video or image fields)

    Args:
        example: Dataset example containing video/image/text data
        return_pil: If True, return raw video frames/images (PIL Images)
                   If False, use video/image path/URL directly

    Returns:
        Formatted message log dictionary
    """

    # Collect multimodal content
    multimodal_content = []

    # Handle video content (check both "video" and "video_frames" field names)
    video_value = example.get("video") or example.get("video_frames")

    if video_value is not None:
        # If video_value is a list of strings (frame paths), load them as PIL Images
        if isinstance(video_value, (list, tuple)) and len(video_value) > 0:
            if isinstance(video_value[0], str):
                # Load frame paths into PIL Images
                video_value = [Image.open(frame_path).convert("RGB") for frame_path in video_value]

                # Apply horizontal flip if requested in dataset
                if example.get("need_to_flip", False):
                    video_value = [ImageOps.mirror(frame) for frame in video_value]

        video_content = {
            "type": "video",
            "video": video_value,  # Path, URL, or PIL frames - no encoding needed
        }

        # Add video metadata
        if example.get('fps') is None and 'dataset_type' in example:
            print(example['dataset_type'])
        if "fps" in example or "sample_fps" in example:
            fps_value = example.get("fps", example.get("sample_fps", 10.0))
            if isinstance(fps_value, str):
                fps_value = float(fps_value)
        else:
            fps_value = 10.0

        if isinstance(video_value, (list, tuple)) and len(video_value) > 0:
            # Get frame dimensions from first frame
            first_frame = video_value[0]
            if hasattr(first_frame, 'size'):  # PIL Image
                width, height = first_frame.size
            else:
                height, width = first_frame.shape[-2:]

            video_metadata = VideoMetadata(
                total_num_frames=len(video_value),
                fps=fps_value,
                width=width,
                height=height,
                frames_indices=list(range(len(video_value))),  # All frames, in order
            )
            video_content["video_metadata"] = video_metadata

        if "max_pixels" in example:
            video_content["max_pixels"] = int(example["max_pixels"])
        if "min_pixels" in example:
            video_content["min_pixels"] = int(example["min_pixels"])

        multimodal_content.append(video_content)

    # Handle image content (check both "image" and "images" field names) if no video
    else:
        image_value = example.get("image") or example.get("images")

        if image_value is not None:
            # Ensure images is always a list
            if not isinstance(image_value, (list, tuple)):
                image_value = [image_value]

            # If image_value is a list of strings (image paths), load them as PIL Images
            if len(image_value) > 0 and isinstance(image_value[0], str):
                image_value = [Image.open(img_path).convert("RGB") for img_path in image_value]

                # Apply horizontal flip if requested in dataset
                if example.get("need_to_flip", False):
                    image_value = [ImageOps.mirror(img) for img in image_value]

            # Add each image as separate content item
            for img in image_value:
                image_content = {
                    "type": "image",
                    "image": img,  # PIL Image or path - no encoding needed
                }

                # Add pixel constraints if present
                if "max_pixels" in example:
                    image_content["max_pixels"] = int(example["max_pixels"])
                if "min_pixels" in example:
                    image_content["min_pixels"] = int(example["min_pixels"])

                multimodal_content.append(image_content)

    # If messages are already formatted in the dataset, use them directly
    if "messages" in example:
        # Find the user message and inject the multimodal content (if any)
        messages = example["messages"]
        for msg in messages:
            if msg["role"] == "user":
                # If content is a string, convert to list format
                if isinstance(msg["content"], str):
                    # Prepend multimodal content (videos/images) before text
                    msg["content"] = multimodal_content + [
                        {"type": "text", "text": msg["content"]}
                    ]
                # If content is already a list, prepend multimodal content
                elif isinstance(msg["content"], list):
                    msg["content"] = multimodal_content + msg["content"]
                break

        ret = {
            "messages": messages,
            "task_name": example.get("task_name", "thrive-vlm"),
        }
    else:
        # Original format: build messages from question/answer fields
        # Add multimodal content first, then text
        user_content = multimodal_content + [
            {
                "type": "text",
                "text": str(example["question"]),
            },
        ]

        assistant_content = str(example["answer"]).strip()

        ret = {
            "messages": [
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": assistant_content,
                },
            ],
            "task_name": "thrive-vlm",
        }

    return ret


def prepare_thrive_vlm_dataset(
    split: str = "train",
    dataset_name: str = "thrive-vlm",
    task_name: Optional[str] = None,
):
    """Prepare THRIVE-VLM dataset for training.

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
    if hasattr(raw, 'keys') and callable(raw.keys):
        # It's a DatasetDict with splits
        if split == "train":
            train_dataset = raw["train"]
            val_dataset = raw.get("validation", raw.get("val", raw["train"]))
        else:
            if split not in raw:
                raise ValueError(f"Split '{split}' not found. Available: {list(raw.keys())}")

            train_dataset = raw[split]
            val_dataset = raw[split]
    else:
        # It's a single Dataset, use it for both train and validation
        train_dataset = raw
        val_dataset = raw

    # Format - disable features to avoid schema conflicts
    train_dataset = train_dataset.add_column("task_name", [task_name] * len(train_dataset))
    val_dataset = val_dataset.add_column("task_name", [task_name] * len(val_dataset))

    return {
        "train": train_dataset,
        "validation": val_dataset,
    }


class ThriveVLMDataset:
    """Dataset class for THRIVE-VLM video question answering.

    This dataset handles video inputs with question-answer pairs for VLM training.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        data_path: Optional[str] = None,
        **kwargs,  # Accept other params from config
    ):
        if split not in ["train", "validation"]:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train' or 'validation'."
            )
        self.task_name = "thrive-vlm"

        # Use data_path if provided, otherwise use dataset_name as the path
        path = data_path if data_path is not None else dataset_name

        self.formatted_ds = prepare_thrive_vlm_dataset(
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
        """Set the data processor to sft_processor.

        For Thrive VLM, we use sft_processor which will be configured
        with format_thrive_vlm_dataset as datum_preprocessor in run_sft.py.
        """
        from nemo_rl.data.processors import sft_processor

        self.processor = sft_processor
