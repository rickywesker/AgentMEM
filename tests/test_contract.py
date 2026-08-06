"""Contract tests.

A 400/422 from us is not retried by the platform — it aborts the run. These
tests cover the shapes that would cause that, plus the retrieval invariants
the competition rules make non-negotiable.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

import numpy as np
import pytest

from agentmem import ingest, retrieve
from agentmem.schemas import AddRequest, AddResponse, Memory, SearchRequest, SearchResponse
from agentmem.store import Record, memory_id


def make_request(**overrides) -> AddRequest:
    payload = {
        "request_id": "eval:r1:locomo_refined:conv-0:chunk-0",
        "user_id": "eval:r1:locomo:conv-0",
        "session_id": "eval:r1:sample:0",
        "messages": [
            {"role": "user", "content": "Caroline: I adopted a dog.", "timestamp": 1683525360000},
            {"role": "user", "content": "Melanie: Her name?", "timestamp": 1683525420000},
        ],
    }
    payload.update(overrides)
    return AddRequest(**payload)


class TestAddContract:
    def test_extra_platform_fields_are_tolerated(self):
        request = make_request(trace_id="abc", mode="full")
        assert request.request_id.endswith("chunk-0")

    def test_message_timestamp_is_optional(self):
        request = make_request(messages=[{"role": "user", "content": "hi"}])
        assert request.messages[0].timestamp is None

    def test_response_echoes_every_id(self):
        request = make_request()
        response = AddResponse(
            success=True,
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        body = response.model_dump()
        assert body["success"] is True
        assert isinstance(body["success"], bool)
        for field in ("request_id", "user_id", "session_id"):
            assert body[field] == getattr(request, field)

    def test_rows_are_idempotent_on_request_id(self):
        first = ingest.message_rows(make_request())
        second = ingest.message_rows(make_request())
        assert [row[0] for row in first] == [row[0] for row in second]

    def test_blank_messages_are_dropped_not_stored(self):
        request = make_request(
            messages=[
                {"role": "user", "content": "   ", "timestamp": 1683525360000},
                {"role": "user", "content": "real", "timestamp": 1683525420000},
            ]
        )
        rows = ingest.message_rows(request)
        assert len(rows) == 1
        assert rows[0][5].endswith("real")

    def test_memory_id_is_stable_and_unique_per_ordinal(self):
        assert memory_id("r", "message", 0) == memory_id("r", "message", 0)
        assert memory_id("r", "message", 0) != memory_id("r", "message", 1)
        assert memory_id("r", "message", 0) != memory_id("r", "fact", 0)


class TestSearchContract:
    def test_top_k_ceiling_is_enforced(self):
        with pytest.raises(ValueError):
            SearchRequest(query="q", user_id="u", top_k=101)

    def test_options_absent_for_open_questions(self):
        request = SearchRequest(query="q", user_id="u")
        assert request.options is None

    def test_empty_result_still_has_a_data_array(self):
        assert SearchResponse(data=[]).model_dump() == {"data": []}

    def test_memory_serialises_required_fields(self):
        body = Memory(id="mem_1", content="text", score=0.5).model_dump()
        assert body["id"] and body["content"]
        assert set(body) == {"id", "content", "score", "created_at"}


class TestTimeRendering:
    """The judge compares time granularity exactly, so what we inline matters."""

    def test_date_is_inlined_at_day_granularity(self):
        when = datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
        assert ingest.render("Caroline adopted a dog.", when) == (
            "[08 May 2023] Caroline adopted a dog."
        )

    def test_no_clock_time_leaks_into_content(self):
        when = datetime(2023, 5, 8, 13, 56, 42, tzinfo=UTC)
        rendered = ingest.render("x", when)
        assert "13:56" not in rendered and ":" not in rendered

    def test_undated_content_is_left_alone(self):
        assert ingest.render("no date", None) == "no date"

    def test_anchor_is_the_earliest_timestamp_in_the_chunk(self):
        request = make_request(
            messages=[
                {"role": "user", "content": "b", "timestamp": 1683525420000},
                {"role": "user", "content": "a", "timestamp": 1683525360000},
            ]
        )
        anchor = ingest.chunk_anchor(request)
        assert anchor == ingest.to_datetime(1683525360000)


def record(identifier: str, content: str, kind: str = "message", ordinal: int = 0) -> Record:
    return Record(
        id=identifier,
        content=content,
        kind=kind,
        created_at=datetime(2023, 5, 8, tzinfo=UTC),
        ordinal=ordinal,
        embedding=None,
    )


class TestRetrieval:
    def test_lexical_match_outranks_unrelated_text(self):
        records = [
            record("a", "Caroline adopted a dog named Biscuit", ordinal=0),
            record("b", "Melanie discussed her painting exhibition", ordinal=1),
            record("c", "the weather was cold that week", ordinal=2),
        ]
        pool = retrieve.candidates(records, "What did Caroline name her dog?", None)
        assert pool[0].record.id == "a"

    def test_returns_far_fewer_than_top_k(self):
        records = [record(str(i), f"memory number {i} about dogs", ordinal=i) for i in range(200)]
        pool = retrieve.candidates(records, "dogs", None)
        assert len(retrieve.trim(pool, limit=retrieve.DEFAULT_RETURN)) <= retrieve.DEFAULT_RETURN

    def test_near_duplicates_are_collapsed(self):
        records = [
            record("a", "Caroline adopted a dog named Biscuit", ordinal=0),
            record("b", "Caroline adopted a dog named Biscuit", ordinal=1),
            record("c", "Melanie painted a sunrise", ordinal=2),
        ]
        pool = retrieve.candidates(records, "Caroline dog Biscuit", None, floor=0)
        # Both copies are candidates; only one survives the trim.
        assert {s.record.id for s in pool} == {"a", "b"}
        assert [s.record.id for s in retrieve.trim(pool, limit=10)] == ["a"]

    def test_backfill_only_reaches_the_floor_not_the_pool(self):
        records = [record(str(i), f"unrelated topic {i}", ordinal=i) for i in range(40)]
        pool = retrieve.candidates(records, "zzzz qqqq", None, floor=5)
        assert len(pool) == 5

    def test_backfill_prefers_recent_records(self):
        records = [record(str(i), f"unrelated topic {i}", ordinal=i) for i in range(40)]
        pool = retrieve.candidates(records, "zzzz qqqq", None, floor=3)
        assert {s.record.id for s in pool} == {"39", "38", "37"}

    def test_strong_match_is_not_padded_out(self):
        records = [record("hit", "Caroline adopted a dog named Biscuit", ordinal=0)]
        records += [record(str(i), f"unrelated topic {i}", ordinal=i + 1) for i in range(30)]
        kept = retrieve.trim(retrieve.candidates(records, "Biscuit", None), limit=20)
        assert kept[0].record.id == "hit"
        # The whole thesis: one strong match must not drag in 19 distractors.
        assert len(kept) <= retrieve.MIN_CANDIDATES

    def test_slot_quota_keeps_both_record_kinds(self):
        """Facts must not starve raw turns: temporal and unanswerable
        questions are carried by the dated verbatim turns."""
        scored = [
            retrieve.Scored(record=record(f"f{i}", f"fact {i}", "fact"), score=1.0 - i / 100)
            for i in range(20)
        ] + [
            retrieve.Scored(record=record(f"m{i}", f"message {i}", "message"), score=0.5 - i / 100)
            for i in range(20)
        ]
        kept = retrieve.trim(scored, limit=10, fact_share=0.5)
        kinds = Counter(item.record.kind for item in kept)
        assert kinds["fact"] == 5 and kinds["message"] == 5

    def test_quota_slack_goes_to_the_other_kind(self):
        scored = [
            retrieve.Scored(record=record("f0", "only fact", "fact"), score=0.9)
        ] + [
            retrieve.Scored(record=record(f"m{i}", f"message {i}", "message"), score=0.5 - i / 100)
            for i in range(20)
        ]
        kept = retrieve.trim(scored, limit=10, fact_share=0.5)
        assert len(kept) == 10
        assert Counter(i.record.kind for i in kept)["fact"] == 1

    def test_fact_share_zero_returns_only_messages(self):
        scored = [
            retrieve.Scored(record=record("f0", "a fact", "fact"), score=0.9),
            retrieve.Scored(record=record("m0", "a message", "message"), score=0.1),
        ]
        kept = retrieve.trim(scored, limit=5, fact_share=0.0)
        assert [i.record.kind for i in kept] == ["message"]

    def test_char_budget_caps_total_returned_text(self):
        items = [
            retrieve.Scored(
                record=record(str(i), f"record {i} " + f"word{i} " * 120), score=1 - i / 100
            )
            for i in range(50)
        ]
        kept = retrieve.trim(items, limit=50, fact_share=0.0, char_budget=10_000)
        assert sum(len(k.record.content) for k in kept) <= 10_000
        assert len(kept) < 50

    def test_char_budget_keeps_the_most_relevant(self):
        items = [
            retrieve.Scored(
                record=record(str(i), f"record {i} " + f"word{i} " * 120), score=1 - i / 100
            )
            for i in range(50)
        ]
        kept = retrieve.trim(items, limit=50, fact_share=0.0, char_budget=5_000)
        assert kept[0].record.id == "0"

    def test_char_budget_never_returns_nothing(self):
        """An empty context scores 10.4 against 63.4, so one over-long
        record still beats returning none."""
        items = [retrieve.Scored(record=record("big", "x " * 5000), score=1.0)]
        assert len(retrieve.trim(items, limit=10, fact_share=0.0, char_budget=10)) == 1

    def test_char_budget_zero_disables_the_cap(self):
        items = [
            retrieve.Scored(
                record=record(str(i), f"record {i} " + f"word{i} " * 120), score=1 - i / 100
            )
            for i in range(30)
        ]
        assert len(retrieve.trim(items, limit=30, fact_share=0.0, char_budget=0)) == 30

    def test_output_stays_in_relevance_order(self):
        scored = [
            retrieve.Scored(record=record("m0", "message zero", "message"), score=0.9),
            retrieve.Scored(record=record("f0", "fact zero", "fact"), score=0.4),
        ]
        kept = retrieve.trim(scored, limit=5, fact_share=0.5)
        assert [i.score for i in kept] == sorted((i.score for i in kept), reverse=True)
        assert kept[0].record.id == "m0"

    def test_no_signal_still_returns_candidates(self):
        records = [record(str(i), f"unrelated content {i}", ordinal=i) for i in range(5)]
        assert retrieve.candidates(records, "zzzz qqqq", None)

    def test_empty_corpus_is_not_an_error(self):
        assert retrieve.candidates([], "anything", None) == []

    def test_options_widen_lexical_recall(self):
        records = [
            record("a", "Melanie took up watercolour painting", ordinal=0),
            record("b", "Caroline started running marathons", ordinal=1),
        ]
        without = retrieve.candidates(records, "What is the hobby?", None)
        with_options = retrieve.candidates(
            records, "What is the hobby?", None, options=["A. watercolour", "B. cooking"]
        )
        assert with_options[0].record.id == "a"
        assert {s.record.id for s in without} == {"a", "b"}

    def test_dense_and_lexical_signals_fuse(self):
        records = [
            record("a", "completely different words", ordinal=0),
            record("b", "dog adoption story", ordinal=1),
        ]
        records[0].embedding = np.array([1.0, 0.0], dtype=np.float32)
        records[1].embedding = np.array([0.0, 1.0], dtype=np.float32)
        pool = retrieve.candidates(
            records, "dog", np.array([1.0, 0.0], dtype=np.float32)
        )
        # Both surface: lexical favours "b", dense favours "a".
        assert {s.record.id for s in pool} == {"a", "b"}

    def test_scores_are_monotonically_non_increasing(self):
        records = [record(str(i), f"dog story {i}", ordinal=i) for i in range(10)]
        pool = retrieve.candidates(records, "dog story", None)
        scores = [s.score for s in pool]
        assert scores == sorted(scores, reverse=True)


class TestIsolation:
    """user_id is the only retrieval boundary; crossing it is disqualifying."""

    def test_search_payload_carries_no_user_identifiers(self):
        payload = ingest.as_search_payload(record("mem_1", "text"))
        assert set(payload) == {"id", "content", "created_at"}
