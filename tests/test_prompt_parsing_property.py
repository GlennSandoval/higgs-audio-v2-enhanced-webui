"""Property-based tests for multi-speaker prompt utilities."""

from __future__ import annotations

import string

import pytest

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
given = hypothesis.given

from app.services.generation_service import GenerationService

printable_letters = string.ascii_letters + string.digits + " .,!?"  # simple alphabet for prompts
sentence_strategy = st.text(alphabet=printable_letters, min_size=1, max_size=80)
speaker_strategy = st.text(alphabet=string.ascii_uppercase, min_size=1, max_size=5)


@given(st.lists(sentence_strategy, min_size=1, max_size=5))
def test_auto_format_and_parse_round_trip(sentences: list[str]) -> None:
    raw = "\n".join(sentences)
    formatted = GenerationService._auto_format_multi_speaker(raw)

    assert "[SPEAKER" in formatted

    parsed = GenerationService._parse_multi_speaker_text(formatted)
    combined = "\n".join("\n".join(chunks) for chunks in parsed.values())

    for sentence in sentences:
        assert sentence.strip() in combined


@given(
    st.lists(
        st.tuples(speaker_strategy, sentence_strategy),
        min_size=1,
        max_size=5,
        unique_by=lambda pair: pair[0],
    )
)
def test_convert_to_speaker_format_preserves_mapping(pairs: list[tuple[str, str]]) -> None:
    mapping = {name: f"SPEAKER{index}" for index, (name, _) in enumerate(pairs)}
    raw_lines = [f"[{name}] {utterance}" for name, utterance in pairs]
    raw_text = "\n".join(raw_lines)

    converted = GenerationService._convert_to_speaker_format(raw_text, mapping)
    parsed = GenerationService._parse_multi_speaker_text(converted)

    expected_keys = {f"SPEAKER{index}" for index in range(len(pairs))}
    assert set(parsed.keys()) == expected_keys

    expected_counts: dict[str, int] = {}
    for name, _utterance in pairs:
        replacement = mapping[name]
        expected_counts[replacement] = expected_counts.get(replacement, 0) + 1

    for replacement, count in expected_counts.items():
        assert len(parsed.get(replacement, [])) == count
