"""Prompt templates for Gemini AI calls.

All prompts live here so they can be reviewed and tweaked in one place.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt for audio → structured form extraction
# ---------------------------------------------------------------------------

FILL_SYSTEM_PROMPT = """\
You are FormSaathi, an AI assistant that helps fill bank account opening forms \
from spoken input. You listen to what the customer says and extract form field \
values accurately.

CRITICAL RULES:
1. ONLY extract values the person actually said. If a field was NOT mentioned, \
set its value to null. NEVER guess or invent data.
2. NEVER extract or ask for Aadhaar numbers, PAN numbers, or any government ID \
numbers. The form only records which type of ID proof is attached, not the number.
3. Handle Tamil, English, and mixed Tamil-English speech.
4. For dates, convert to DD-MM-YYYY format.
5. For phone numbers, extract exactly the digits spoken. Indian mobile numbers \
are 10 digits starting with 6-9.
6. For names, capitalize properly.
7. For select/choice fields, map spoken answers to the closest allowed option value.
8. Assign a confidence level to each extracted field:
   - HIGH: clearly and unambiguously spoken
   - MEDIUM: likely correct but could have alternatives
   - LOW: unclear, partially audible, or uncertain

This is a demo application. Use only the fields defined below.
"""


def build_fill_user_prompt(
    language: str,
    field_descriptions: str,
) -> str:
    """Build the user prompt for the /api/fill Gemini call.

    Args:
        language: Language code (``en`` or ``ta``).
        field_descriptions: Formatted string of field IDs, labels, and types.

    Returns:
        The user prompt string.
    """
    lang_name = "Tamil" if language == "ta" else "English"
    return (
        f"The customer is speaking in {lang_name} (possibly mixed with English). "
        f"Listen to the audio and extract the following form fields:\n\n"
        f"{field_descriptions}\n\n"
        f"Return a JSON object with 'transcript' (the full transcription of what "
        f"was said), 'language_detected' ('en' or 'ta'), and 'fields' (an object "
        f"where each key is the field id and each value has 'value' (string or "
        f"null) and 'confidence' ('low', 'medium', or 'high'))."
    )


def format_field_descriptions(fields: list[dict]) -> str:
    """Format form field definitions for the AI prompt.

    Args:
        fields: List of field dicts with id, label_en, label_ta, type, options.

    Returns:
        A human-readable field description string.
    """
    lines: list[str] = []
    for f in fields:
        line = f"- {f['id']}: {f['label_en']} / {f['label_ta']} (type: {f['type']})"
        if f.get("options"):
            opts = ", ".join(
                f"{o['value']}={o['label_en']}" for o in f["options"]
            )
            line += f" [options: {opts}]"
        if f.get("required"):
            line += " [REQUIRED]"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt for field explanations
# ---------------------------------------------------------------------------

EXPLAIN_SYSTEM_PROMPT = """\
You are FormSaathi, a helpful assistant that explains bank account form fields \
in simple words that anyone can understand.

RULES:
1. Only explain form fields. If asked about anything else (financial advice, \
investment tips, loan details, personal opinions, unrelated topics), politely \
refuse and say you can only help with form fields.
2. Use very simple, everyday language. Imagine explaining to someone who has \
never seen a bank form before.
3. Keep answers to at most 2 short sentences.
4. If the language is Tamil (ta), respond in Tamil. If English (en), respond \
in English.
5. Use examples when helpful.
6. Never mention or ask for ID numbers.
"""


def build_explain_prompt(
    field_label: str,
    field_help: str,
    language: str,
    user_question: str | None = None,
) -> str:
    """Build the user prompt for the /api/explain Gemini call.

    Args:
        field_label: The label of the field being asked about.
        field_help: The built-in help text for the field.
        language: Language code (``en`` or ``ta``).
        user_question: Optional spoken/typed question from the user.

    Returns:
        The user prompt string.
    """
    lang_name = "Tamil" if language == "ta" else "English"
    prompt = (
        f"The user needs help understanding the form field: \"{field_label}\".\n"
        f"Built-in help: \"{field_help}\"\n"
        f"Respond in {lang_name}."
    )
    if user_question:
        prompt += f"\nThe user asks: \"{user_question}\""
    else:
        prompt += "\nExplain this field simply."
    return prompt
