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

"""Rubric templates for LLM-as-a-judge in Thrive VLM environment."""

from typing import Any


# =============================================================================
# Rubric Templates
# =============================================================================

REPETITION_ANALYSIS_RUBRIC = """================================================================================
LLM-as-Judge Prompt for GRPO - Repetition Analysis
================================================================================
Purpose: Single judge call to evaluate feedback tone, informativeness,
         feedback-severity agreement, and analysis-severity agreement
================================================================================

You are evaluating the output of a physiotherapy AI assistant that analyzed a member's exercise form. 
Here is the model's analysis:

Ground truth error severity scores (1 = not present, 6 = severe):
{error_scores}

Model's movement analysis:
{movement_analysis}

Model's therapist feedback:
{feedback}

---

Score each of the following rubrics independently on a scale of 1-5.

RUBRIC 1 — FEEDBACK TONE
How warm, supportive, and encouraging does the feedback sound?

1 = Cold, clinical, or robotic. Sounds like a system message or textbook.
2 = Neutral but impersonal. Correct but no warmth.
3 = Polite but formulaic. Some encouraging language but feels templated.
4 = Warm and natural. Sounds like a real person. Encouraging without being over the top.
5 = Excellent. Natural, conversational, genuinely supportive. The member would feel motivated.

RUBRIC 2 — FEEDBACK INFORMATIVENESS
Does the feedback give the member a clear, actionable cue they can apply on the next repetition?

1 = No useful information. Vague praise or criticism with nothing actionable.
2 = Identifies a problem area but no actionable cue.
3 = Gives a general direction but lacks specificity.
4 = Clear and actionable. The member knows exactly what to change.
5 = Highly specific and actionable, targeting the most important issue.

RUBRIC 3 — FEEDBACK-SEVERITY AGREEMENT
Does the feedback address the most significant errors? If all errors are severity 1, does the feedback acknowledge good form?

1 = Feedback contradicts the errors. Praises something scored high, or corrects something scored 1.
2 = Feedback addresses a minor error while major errors are ignored.
3 = Feedback addresses a real error but not one of the most important ones.
4 = Feedback addresses one of the most significant errors.
5 = Feedback directly and specifically addresses the most significant errors.

RUBRIC 4 — MOVEMENT ANALYSIS-SEVERITY AGREEMENT
Is the movement analysis consistent with the error severity scores? High severity errors should be prominently discussed, errors scored 1 should be noted as absent or barely mentioned.

1 = Major contradictions. Analysis describes errors as severe but scored 1, or movement as correct but errors scored 4+.
2 = Notable inconsistencies. One or two errors described at a level that doesn't match their score.
3 = Mostly consistent with minor mismatches.
4 = Consistent. High severity errors get prominent discussion, low severity errors get brief mention.
5 = Fully consistent. Every error's discussion corresponds to its severity level.

---

Respond with only the four scores in this exact format:

<tone>X</tone>
<informativeness>X</informativeness>
<feedback_agreement>X</feedback_agreement>
<analysis_agreement>X</analysis_agreement>
"""

RUBRIC_REGISTRY = {
    "repetition_analysis": REPETITION_ANALYSIS_RUBRIC,
    "movement_assessment": "",  # TODO
    "error_detection": "",  # TODO
    "two_stage": "",  # TODO
    "full_exercise": "",  # TODO
}


def _extract_section(response: str, section_name: str) -> str:
    """Extract content between [SECTION_NAME] markers.

    Args:
        response: Full model response
        section_name: Name of section (e.g., "MOVEMENT ANALYSIS", "FEEDBACK")

    Returns:
        Content of the section, or empty string if not found
    """
    import re

    # Match [SECTION_NAME]\n content \n[NEXT_SECTION or end]
    pattern = rf'\[{section_name}\]\s*\n(.*?)(?=\[|$)'
    match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)

    if match:
        return match.group(1).strip()
    return ""


def format_error_scores_for_judge(gt_errors: dict[str, float]) -> str:
    """Format error scores as simple list for judge prompt.

    Args:
        gt_errors: Dictionary of error_name -> severity_score

    Returns:
        Formatted string like "Error Name 1: 3\nError Name 2: 1"
    """
    if not gt_errors:
        return "(No errors detected)"
    return "\n".join(f"{name}: {int(score)}" for name, score in gt_errors.items())


def get_rubric(rubric_type: str, custom_rubric_path: str | None = None) -> str:
    """Get a rubric template by type.

    Args:
        rubric_type: Type of rubric ("movement_assessment", "error_detection",
                     "two_stage", "full_exercise", or "custom")
        custom_rubric_path: Path to custom rubric file (required if rubric_type="custom")

    Returns:
        Rubric template string

    Raises:
        ValueError: If rubric_type is invalid or custom_rubric_path is missing
    """
    if rubric_type == "custom":
        if custom_rubric_path is None:
            raise ValueError("custom_rubric_path is required when rubric_type='custom'")
        with open(custom_rubric_path, 'r') as f:
            return f.read()

    if rubric_type not in RUBRIC_REGISTRY:
        available = ", ".join(RUBRIC_REGISTRY.keys())
        raise ValueError(
            f"Unknown rubric_type: {rubric_type}. Available: {available}, custom"
        )

    return RUBRIC_REGISTRY[rubric_type]


def format_judge_prompt(
    rubric: str,
    video_context: str,
    model_response: str,
    ground_truth: str | None = None,
    task_type: str = "repetition",
) -> str:
    """Format the complete judge prompt.

    Args:
        rubric: The rubric template to use
        video_context: Description or context about the video
        model_response: The model's assessment to judge
        ground_truth: Optional ground truth assessment for reference
        task_type: Type of task ("repetition" or "full_exercise")

    Returns:
        Formatted prompt string for the judge
    """
    # Extract sections from model response
    movement_analysis = _extract_section(model_response, "MOVEMENT ANALYSIS")
    feedback = _extract_section(model_response, "FEEDBACK")

    # If sections not found, use fallback
    if not movement_analysis:
        movement_analysis = "(Movement analysis section not found in model response)"
    if not feedback:
        feedback = "(Feedback section not found in model response)"

    # Extract error scores from ground truth
    error_scores_text = ""
    if ground_truth:
        # Extract from [ERRORS] section in ground truth
        gt_errors_section = _extract_section(ground_truth, "ERRORS")
        if gt_errors_section:
            error_scores_text = gt_errors_section
        else:
            # Fallback: try to parse error scores directly
            # This handles old format or plain error score lists
            error_scores_text = "(Error scores not found in ground truth)"
    else:
        error_scores_text = "(Ground truth not provided)"

    # Check if rubric has template variables
    if "{movement_analysis}" in rubric and "{feedback}" in rubric and "{error_scores}" in rubric:
        # Format rubric with extracted sections
        try:
            formatted_rubric = rubric.format(
                movement_analysis=movement_analysis,
                feedback=feedback,
                error_scores=error_scores_text
            )
            return formatted_rubric
        except KeyError as e:
            # If formatting fails, return rubric as-is with note
            return f"{rubric}\n\n[Note: Template formatting failed: {e}]"
    else:
        # Rubric doesn't use templates, return as-is
        return rubric


def parse_judge_output(judge_response: str) -> dict[str, Any]:
    """Parse judge output to extract scores.

    Args:
        judge_response: Raw response from judge model

    Returns:
        Dictionary with parsed scores and reasoning. Must include "total_reward" key.

    Raises:
        ValueError: If output cannot be parsed
    """
    import json
    import re

    # Remove thinking trace (between <think> and </think>) to avoid parsing scores from reasoning
    judge_response = re.sub(r'<think>.*?</think>', '', judge_response, flags=re.DOTALL | re.IGNORECASE)

    # Try to parse XML-like tags first (for repetition_analysis rubric)
    tone_match = re.search(r'<tone>(\d+)</tone>', judge_response, re.IGNORECASE)
    info_match = re.search(r'<informativeness>(\d+)</informativeness>', judge_response, re.IGNORECASE)
    feedback_match = re.search(r'<feedback_agreement>(\d+)</feedback_agreement>', judge_response, re.IGNORECASE)
    analysis_match = re.search(r'<analysis_agreement>(\d+)</analysis_agreement>', judge_response, re.IGNORECASE)

    if all([tone_match, info_match, feedback_match, analysis_match]):
        # Repetition analysis format with 4 rubrics (1-5 scale)
        scores = [
            int(tone_match.group(1)),
            int(info_match.group(1)),
            int(feedback_match.group(1)),
            int(analysis_match.group(1))
        ]

        # Normalize from 1-5 scale to 0-1 scale
        normalized = [(s - 1) / 4.0 for s in scores]

        return {
            "tone": normalized[0],
            "informativeness": normalized[1],
            "feedback_agreement": normalized[2],
            "analysis_agreement": normalized[3],
            "total_reward": sum(normalized) / 4.0,
            "reasoning": f"Scores (1-5): tone={scores[0]}, informativeness={scores[1]}, feedback_agreement={scores[2]}, analysis_agreement={scores[3]}"
        }

    # Fallback: try JSON format (for other rubrics)
    json_match = re.search(r'```json\s*(.*?)\s*```', judge_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON without code block
        json_match = re.search(r'\{.*\}', judge_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError(f"Could not find scores in judge response: {judge_response[:200]}")

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in judge response: {e}\n{json_str[:200]}")

    # Validate that total_reward exists
    if "total_reward" not in result:
        raise ValueError(f"Judge output missing 'total_reward' field: {result}")

    return result
