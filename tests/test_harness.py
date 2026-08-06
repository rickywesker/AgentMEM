"""Harness correctness.

The harness is the only thing standing between a configuration change and a
submission we cannot retry for three months. A bug here does not fail loudly —
it quietly shifts every score, which is worse.
"""

from __future__ import annotations

from datetime import UTC, datetime

from harness import contracts
from harness.adapters import locomo


class TestSessionTimestamps:
    def test_midnight_keeps_its_own_date(self):
        """Regression: naive strptime().timestamp() reads as machine-local
        time, which pushed every '12:xx am' session back one day on a UTC+8
        host — 17 of LoCoMo's 288 sessions. The judge grades day granularity
        exactly, so that is a silently lost temporal point each time."""
        epoch = locomo._session_epoch_ms("12:09 am on 13 September, 2023")
        assert datetime.fromtimestamp(epoch / 1000, tz=UTC).date() == datetime(
            2023, 9, 13, tzinfo=UTC
        ).date()

    def test_late_evening_keeps_its_own_date(self):
        epoch = locomo._session_epoch_ms("11:30 pm on 8 May, 2023")
        moment = datetime.fromtimestamp(epoch / 1000, tz=UTC)
        assert (moment.date(), moment.hour) == (datetime(2023, 5, 8).date(), 23)

    def test_afternoon_hour_is_converted_from_twelve_hour_clock(self):
        epoch = locomo._session_epoch_ms("1:56 pm on 8 May, 2023")
        assert datetime.fromtimestamp(epoch / 1000, tz=UTC).hour == 13

    def test_noon_and_midnight_do_not_collide(self):
        noon = locomo._session_epoch_ms("12:00 pm on 8 May, 2023")
        midnight = locomo._session_epoch_ms("12:00 am on 8 May, 2023")
        assert datetime.fromtimestamp(noon / 1000, tz=UTC).hour == 12
        assert datetime.fromtimestamp(midnight / 1000, tz=UTC).hour == 0

    def test_unparseable_stamp_is_none_not_an_exception(self):
        assert locomo._session_epoch_ms("sometime last week") is None
        assert locomo._session_epoch_ms("") is None


class TestChunking:
    def test_splits_at_twenty_messages(self):
        messages = [{"content": "a word", "role": "user"} for _ in range(45)]
        sizes = [len(c) for c in locomo._chunk_messages(messages)]
        assert sizes == [20, 20, 5]

    def test_splits_early_when_the_word_budget_trips_first(self):
        messages = [{"content": "word " * 600, "role": "user"} for _ in range(6)]
        chunks = locomo._chunk_messages(messages)
        assert all(len(c) < locomo.CHUNK_MESSAGES for c in chunks)
        assert len(chunks) > 1

    def test_no_message_is_dropped_or_duplicated(self):
        messages = [{"content": f"m{i}", "role": "user"} for i in range(53)]
        flattened = [m for chunk in locomo._chunk_messages(messages) for m in chunk]
        assert [m["content"] for m in flattened] == [f"m{i}" for i in range(53)]


class TestDatasetLoad:
    def test_ids_are_unique_across_chunks_and_questions(self):
        chunks, questions = locomo.load("harness/datasets/locomo/data/locomo10.json")
        assert len({c.request_id for c in chunks}) == len(chunks)
        assert len({q.id for q in questions}) == len(questions)

    def test_every_question_maps_to_an_ingested_user(self):
        chunks, questions = locomo.load("harness/datasets/locomo/data/locomo10.json")
        assert {q.user_id for q in questions} <= {c.user_id for c in chunks}

    def test_adversarial_questions_use_their_own_gold_field(self):
        _, questions = locomo.load("harness/datasets/locomo/data/locomo10.json")
        adversarial = [q for q in questions if q.category == 5]
        assert adversarial and all(q.gold for q in adversarial)


class TestPlatformContracts:
    """These strings are copied from the platform. Drift is the failure mode."""

    def test_answer_prompt_keeps_the_instructions_we_optimise_against(self):
        template = contracts.OPEN_ENDED_ANSWER_TEMPLATE
        assert "Preserve specific names, titles, places, and labels" in template
        assert "include all required items and no extras" in template
        assert "Answer with the shortest correct phrase or sentence" in template

    def test_retrieved_context_lands_in_the_speaker_one_slot(self):
        rendered = contracts.render_answer_prompt(
            {"question": "Q?", "retrieved_context": "MEMORY-BLOCK", "speaker_1_name": "Caroline"}
        )
        assert "Memories for user Caroline:\n\nMEMORY-BLOCK" in rendered
        assert "{{" not in rendered

    def test_judge_prompt_interpolates_all_three_fields(self):
        rendered = contracts.render_accuracy_prompt(
            {"question": "When?", "gold_answer": "7 May 2023"}, "8 May 2023"
        )
        assert "When?" in rendered and "7 May 2023" in rendered and "8 May 2023" in rendered
        # The JSON example block must survive interpolation intact.
        assert '"label": "CORRECT" or "WRONG"' in rendered

    def test_judge_label_parsing(self):
        assert contracts.parse_judge_label('```json\n{"label": "CORRECT"}\n```') == "CORRECT"
        assert contracts.parse_judge_label('reasoning… {"label":"wrong"}') == "WRONG"

    def test_unparseable_judge_output_raises_rather_than_scoring(self):
        for bad in ("no json here", '{"label": "MAYBE"}'):
            try:
                contracts.parse_judge_label(bad)
            except ValueError:
                continue
            raise AssertionError(f"{bad!r} should not have produced a label")
