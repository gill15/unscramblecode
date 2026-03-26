SYSTEM_PROMPT = """You are UnscrambleCode, a code restoration model.
You receive SCRAMBLED source code (with potentially reordered blocks, renamed identifiers, broken formatting, duplicated lines, missing imports, or extra noise).

Goal: output the CLEAN, CANONICAL, UNSCRAMBLED code for the same program.

Rules:
- Preserve behavior.
- Prefer minimal edits needed to restore correct, idiomatic code.
- Do not add commentary.
- Output ONLY code (no markdown fences, no explanations).
"""


def make_user_content(language: str, scrambled: str) -> str:
    return f"Language: {language}\n\nSCRAMBLED:\n{scrambled}\n\nUNSCRAMBLED:\n"


def make_messages(language: str, scrambled: str, clean: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_content(language, scrambled)},
    ]
    if clean is not None:
        msgs.append({"role": "assistant", "content": clean})
    return msgs


def make_training_example(language: str, scrambled: str, clean: str) -> dict:
    prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"Language: {language}\n\n"
        f"SCRAMBLED:\n{scrambled}\n\n"
        f"UNSCRAMBLED:\n"
    )
    return {
        "language": language,
        "prompt": prompt,
        "response": clean,
        "messages": make_messages(language, scrambled, clean),
    }

