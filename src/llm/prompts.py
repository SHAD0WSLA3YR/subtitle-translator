"""Prompt templates for Japanese-to-English translation refinement.

The LLM refiner improves on Whisper's raw translation output,
fixing grammar, improving fluency, and keeping output concise
for subtitle display.
"""

SYSTEM_PROMPT = """\
You are a Japanese-to-English translation refinement assistant.
You will receive a raw machine translation of Japanese speech.
Your job is to:
1. Fix any grammar issues
2. Make it natural English
3. Keep the original meaning intact
4. Keep it concise (suitable for subtitles)

Output ONLY the refined translation, no explanations."""


def build_messages(raw_translation: str) -> list[dict]:
    """Build chat messages for the LLM refinement request.

    Args:
        raw_translation: Raw output from faster-whisper translate.

    Returns:
        List of {"role": ..., "content": ...} dicts for the chat API.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Refine this translation:\n{raw_translation}"},
    ]
