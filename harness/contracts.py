"""Verbatim answer and judge contracts from the evaluation platform.

Source: https://github.com/AML-memory/agent-memory-leaderboard
  locomo-refined/pipeline.py, longmemeval-s/pipeline.py, scriptmem/pipeline.py

These strings are copied character-for-character on purpose. They are the
scoring function we are optimising against; paraphrasing them silently
invalidates every offline measurement. Do not reformat, do not "fix" the typos
or the curly quotes — the platform has them too.

Platform-fixed parameters (README.md, "Disclosed production parameters"):
  - Answer: gpt-4o-mini, temperature 0, no max_tokens sent, 6 attempts
  - Retrieval: top_k = 100
  - Scores computed on [0, 1], published on [0, 100]
"""

from __future__ import annotations

import json
import re

# --------------------------------------------------------------------------
# Open-ended answer prompt. Shared by locomo-refined and longmemeval-s.
# --------------------------------------------------------------------------
OPEN_ENDED_ANSWER_TEMPLATE = """You are asked to answer a question based on your memories of a conversation.

<instructions>
1. Use only the provided memories. Prefer the memory that answers the question most directly.
2. Your memories are episodic raw observations. Reason about what they imply. Do not refuse just because the answer is not stated verbatim.
3. The question may contain typos. Match it to the most relevant memory even if the wording differs.
4. When multiple answers are possible, list all supported answers, not just the first.
5. For counts or time intervals, enumerate carefully before answering.
6. Preserve specific names, titles, places, and labels from the memories. Use "Rob" not "a colleague", "Sweden" not "home country".
7. Convert relative times like "yesterday", "last month", and "last year" into dates, months, or years when the memory timestamp makes it clear. Keep week-based expressions relative.
8. If memories conflict, prefer the most recent supported memory.
9. For list questions, include all required items and no extras.
10. Keep the final answer minimal. Do not add explanation, background, or extra dates unless needed for correctness.
</instructions>

<memories>
Memories for user {{speaker_1_name}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_name}}:

{{speaker_2_memories}}
</memories>

Question: {{question}}
Answer with the shortest correct phrase or sentence. No preamble, no fluff:"""


# --------------------------------------------------------------------------
# Binary judge. Shared by locomo-refined and longmemeval-s.
# --------------------------------------------------------------------------
ACCURACY_PROMPT = """Your task is to label an answer as ’CORRECT’ or ’WRONG’ given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle — Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold’s key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT — even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold’s content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR.
  Do not answer a gold at a different time unit — even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] → WRONG)
- Do NOT convert relative ↔ absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., “the/last/previous/just prior”) as equivalent when both the anchor date and the time unit are the same.

- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover **all** of them.
- Extra non-contradictory items **generally count as WRONG**.
    - Example: gold = A, B, C ; gen = A, B, C → CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D → WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C → C, C′), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include **any one** of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
```"""


# --------------------------------------------------------------------------
# Multiple-choice answer prompt. ScriptMem.
# --------------------------------------------------------------------------
CHOICE_ANSWER_TEMPLATE = """
You are asked to answer a multiple-choice question based on your memories of a conversation.

<instructions>
1. Use only the provided memories. Prefer the memories that answer the question most directly.
2. Your memories are episodic raw observations. Reason about what they imply. Do not refuse just because the answer is not stated verbatim.
3. The question may contain typos. Match it to the most relevant memories even if the wording differs.
4. The question may be single-choice, multi-select, or ordering. The required output format is different for each type. Obey the format stated in the question, not a default format you invent.
5. Do not add unsupported options. Do not omit supported options. Do not hedge.
6. Preserve option letters exactly. Do not rewrite the answer as option text unless the question explicitly asks for that.
7. If memories conflict, prefer the most recent supported memory.
8. Choose "Cannot infer" only when no memory contains any relevant evidence after scanning all memories. Partial or indirect evidence requires a supported answer, not a refusal.
9. When memories conflict in direction, prefer the one semantically closest to the question's core—not the one with the highest keyword overlap.
10. For multi-select: check every option independently. Include if any memory supports it; exclude only if clearly contradicted or out of scope.
11. For "most plausible", "underlying", or "most strongly implies" questions: compare the top candidates directly before choosing; do not default to the option with the most shared vocabulary.
12. Keep reasoning internal. The visible output must be just the answer string required by the question.
</instructions>

<memories>
Memories for user {{speaker_1_name}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_name}}:

{{speaker_2_memories}}
</memories>

Question: {{question}}
Return only the answer, exactly in the format requested by the question:
""".strip()


_SLOTS = ("speaker_1_name", "speaker_1_memories", "speaker_2_name", "speaker_2_memories", "question")


def memory_text(value: object) -> str:
    """Platform's coercion of a memory field to a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(memory_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def render_answer_prompt(item: dict, template: str = OPEN_ENDED_ANSWER_TEMPLATE) -> str:
    """Mirror of the platform's ``render_answer_prompt``.

    Note the fallback chain: a record with only ``retrieved_context`` puts the
    whole retrieved block into the speaker-1 slot and leaves speaker 2 empty.
    That is the shape our Search results arrive in.
    """
    fallback = item.get("retrieved_context", item.get("memories", ""))
    values = {
        "speaker_1_name": item.get("speaker_1_name", "speaker 1"),
        "speaker_1_memories": item.get("speaker_1_memories", fallback),
        "speaker_2_name": item.get("speaker_2_name", "speaker 2"),
        "speaker_2_memories": item.get("speaker_2_memories", ""),
        "question": item["question"],
    }
    return re.sub(
        r"\{\{(" + "|".join(_SLOTS) + r")\}\}",
        lambda m: memory_text(values[m.group(1)]),
        template,
    )


def gold_answer(item: dict) -> str:
    for key in ("gold_answer", "golden_answer", "reference_answer", "correct_answer"):
        if key in item:
            return memory_text(item[key])
    raise ValueError(f"record {item.get('id', '<unknown>')} has no gold answer")


def render_accuracy_prompt(item: dict, generated_answer: str) -> str:
    values = {
        "question": memory_text(item["question"]),
        "gold_answer": gold_answer(item),
        "generated_answer": generated_answer,
    }
    return re.sub(
        r"\{(question|gold_answer|generated_answer)\}",
        lambda m: values[m.group(1)],
        ACCURACY_PROMPT,
    )


def parse_judge_label(response: str) -> str:
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if not match:
        raise ValueError("judge response does not contain a JSON object")
    payload = json.loads(match.group(0))
    label = str(payload.get("label", "")).upper()
    if label not in {"CORRECT", "WRONG"}:
        raise ValueError("judge label must be CORRECT or WRONG")
    return label
