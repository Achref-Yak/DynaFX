from transformers import AutoTokenizer

from cognitive_engine.chunker import (
    Chunk,
    PropSpan,
    chunk_text,
    merge_propositions,
)

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")


def _count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


class TestChunkText:
    def test_empty_text(self):
        assert chunk_text("", tokenizer) == []

    def test_short_text(self):
        text = "Hello world. This is a short text."
        chunks = chunk_text(text, tokenizer, max_tokens=512, overlap=128)
        assert len(chunks) == 1
        assert chunks[0].start_char == 0
        assert chunks[0].end_char == len(text)
        assert chunks[0].text == text
        assert chunks[0].offset == 0

    def test_long_text_creates_multiple_chunks(self):
        sentence = "The quick brown fox jumps over the lazy dog. "
        token_count = _count_tokens(sentence)
        num_repeats = (600 // token_count) + 1
        text = sentence * num_repeats
        assert _count_tokens(text) > 512

        chunks = chunk_text(text, tokenizer, max_tokens=50, overlap=10)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.end_char > c.start_char
            assert len(c.text) > 0
            assert len(c.tokens) > 0
            assert len(c.offsets) == len(c.tokens)

    def test_chunk_offsets_contiguous(self):
        sentence = "A series of words that will be chunked across multiple windows. "
        token_count = _count_tokens(sentence)
        num_repeats = (600 // token_count) + 1
        text = sentence * num_repeats

        chunks = chunk_text(text, tokenizer, max_tokens=50, overlap=10)

        for c in chunks:
            valid_offsets = [(s, e) for s, e in c.offsets if e > s]
            assert len(valid_offsets) > 0
            for i, (s, e) in enumerate(valid_offsets):
                assert s >= 0
                assert e > s
                if i > 0:
                    assert valid_offsets[i - 1][1] <= s

    def test_chunk_text_ordering(self):
        sentence = "Chunk ordering must be sequential. "
        token_count = _count_tokens(sentence)
        num_repeats = (600 // token_count) + 1
        text = sentence * num_repeats

        chunks = chunk_text(text, tokenizer, max_tokens=50, overlap=10)
        for i, c in enumerate(chunks):
            assert c.offset == i

    def test_chunk_overlap_contains_shared_chars(self):
        text = (
            "The quick brown fox jumps over the lazy dog near the bank. "
            * 30
        )
        chunks = chunk_text(text, tokenizer, max_tokens=50, overlap=20)
        if len(chunks) >= 2:
            overlap_region = set(range(chunks[0].start_char, chunks[0].end_char)) & set(
                range(chunks[1].start_char, chunks[1].end_char)
            )
            assert len(overlap_region) > 0

    def test_chunk_tokens_match_text(self):
        text = "Simple text for token matching verification. " * 10
        chunks = chunk_text(text, tokenizer, max_tokens=50, overlap=10)
        for c in chunks:
            decoded = tokenizer.decode(c.tokens, skip_special_tokens=True)
            assert decoded in text
            assert c.text in text

    def test_no_special_tokens(self):
        text = "No special tokens should appear in chunk tokens."
        chunks = chunk_text(text, tokenizer, max_tokens=512, overlap=128)
        assert len(chunks) == 1
        for c in chunks:
            assert 0 not in c.tokens


class TestMergePropositions:
    def test_empty_chunks(self):
        assert merge_propositions([], [], "") == []

    def test_no_props_returns_empty(self):
        chunk = Chunk(
            start_char=0, end_char=20, text="Hello world test ",
            tokens=[0, 1, 2, 3], offsets=[(0, 5), (6, 11), (12, 16), (17, 20)],
            offset=0,
        )
        tags = [["O", "O", "O", "O"]]
        assert merge_propositions([chunk], tags, "Hello world test ") == []

    def test_single_span(self):
        offsets = [(0, 5), (6, 11), (12, 16), (17, 20)]
        chunk = Chunk(
            start_char=0, end_char=20, text="Hello world test ",
            tokens=[0, 1, 2, 3], offsets=offsets, offset=0,
        )
        tags = [["O", "B-Prop", "I-Prop", "I-Prop"]]
        result = merge_propositions([chunk], tags, "Hello world test ")
        assert len(result) == 1
        assert result[0].start_char == 6
        assert result[0].end_char == 20
        assert result[0].text == "world test "
        assert result[0].chunk_offsets == [0]

    def test_multiple_spans(self):
        offsets = [(0, 2), (3, 8), (9, 12), (13, 18), (19, 22)]
        chunk = Chunk(
            start_char=0, end_char=22, text="hi world foo bar baz",
            tokens=[0, 1, 2, 3, 4], offsets=offsets, offset=0,
        )
        tags = [["B-Prop", "I-Prop", "O", "B-Prop", "I-Prop"]]
        result = merge_propositions(
            [chunk], tags, "hi world foo bar baz",
            merge_margin_chars=3,
        )
        assert len(result) == 2
        assert result[0].text == "hi world"
        assert result[1].text == "bar baz"

    def test_multiple_spans_merged_by_default(self):
        offsets = [(0, 2), (3, 8), (9, 12), (13, 18), (19, 22)]
        chunk = Chunk(
            start_char=0, end_char=22, text="hi world foo bar baz",
            tokens=[0, 1, 2, 3, 4], offsets=offsets, offset=0,
        )
        tags = [["B-Prop", "I-Prop", "O", "B-Prop", "I-Prop"]]
        result = merge_propositions(
            [chunk], tags, "hi world foo bar baz",
        )
        assert len(result) == 1

    def test_overlapping_chunks_merged(self):
        chunk_a = Chunk(
            start_char=0, end_char=30, text="The cat sat on the mat  ",
            tokens=[0, 1, 2, 3, 4, 5, 6],
            offsets=[(0, 3), (4, 7), (8, 11), (12, 14), (15, 18), (19, 22), (22, 24)],
            offset=0,
        )
        chunk_b = Chunk(
            start_char=15, end_char=40, text="the mat and the dog   ",
            tokens=[0, 1, 2, 3, 4, 5, 6],
            offsets=[(15, 18), (19, 22), (23, 26), (27, 30), (31, 34), (34, 37), (37, 40)],
            offset=1,
        )
        full_text = "The cat sat on the mat and the dog   "
        tags_a = [["O", "O", "O", "O", "B-Prop", "I-Prop", "O"]]
        tags_b = [["B-Prop", "I-Prop", "O", "O", "O", "O", "O"]]
        result = merge_propositions([chunk_a, chunk_b], tags_a + tags_b, full_text)
        assert len(result) >= 1
        first = result[0]
        assert first.start_char <= 15
        assert first.end_char >= 22
        assert 0 in first.chunk_offsets
        assert 1 in first.chunk_offsets

    def test_spans_with_small_gap_merged(self):
        offsets = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19)]
        chunk = Chunk(
            start_char=0, end_char=19, text="the cat sat on mat",
            tokens=[0, 1, 2, 3, 4], offsets=offsets, offset=0,
        )
        tags = [["B-Prop", "I-Prop", "O", "B-Prop", "I-Prop"]]
        result = merge_propositions(
            [chunk], tags, "the cat sat on mat",
            merge_margin_chars=10,
        )
        assert len(result) == 1
        assert result[0].start_char == 0
        assert result[0].end_char == 19

    def test_spans_with_large_gap_not_merged(self):
        offsets = [(0, 3), (4, 7), (8, 11), (12, 15), (16, 19)]
        chunk = Chunk(
            start_char=0, end_char=19, text="the cat sat on mat",
            tokens=[0, 1, 2, 3, 4], offsets=offsets, offset=0,
        )
        tags = [["B-Prop", "I-Prop", "O", "B-Prop", "I-Prop"]]
        result = merge_propositions(
            [chunk], tags, "the cat sat on mat",
            merge_margin_chars=1,
        )
        assert len(result) == 2

    def test_int_tags_map_correctly(self):
        offsets = [(0, 5), (6, 11), (12, 16), (17, 20)]
        chunk = Chunk(
            start_char=0, end_char=20, text="Hello world test ",
            tokens=[0, 1, 2, 3], offsets=offsets, offset=0,
        )
        tags = [[0, 1, 2, 2]]
        result = merge_propositions([chunk], tags, "Hello world test ")
        assert len(result) == 1
        assert result[0].text == "world test "

    def test_bare_i_prop_treated_as_start(self):
        offsets = [(0, 3), (4, 7), (8, 11)]
        chunk = Chunk(
            start_char=0, end_char=11, text="foo bar baz",
            tokens=[0, 1, 2], offsets=offsets, offset=0,
        )
        tags = [["O", "I-Prop", "I-Prop"]]
        result = merge_propositions([chunk], tags, "foo bar baz")
        assert len(result) == 1
        assert result[0].text == "bar baz"

    def test_merge_margin_chars_zero(self):
        offsets = [(0, 3), (4, 7), (8, 11)]
        chunk = Chunk(
            start_char=0, end_char=11, text="foo bar baz",
            tokens=[0, 1, 2], offsets=offsets, offset=0,
        )
        tags = [["B-Prop", "O", "B-Prop"]]
        result = merge_propositions(
            [chunk], tags, "foo bar baz",
            merge_margin_chars=0,
        )
        assert len(result) == 2

    def test_deduplication_same_span_from_two_chunks(self):
        chunk_a = Chunk(
            start_char=0, end_char=15, text="Hello world!!!",
            tokens=[0, 1, 2], offsets=[(0, 5), (6, 11), (12, 15)], offset=0,
        )
        chunk_b = Chunk(
            start_char=0, end_char=15, text="Hello world!!!",
            tokens=[3, 4, 5], offsets=[(0, 5), (6, 11), (12, 15)], offset=1,
        )
        full = "Hello world!!!"
        tags_a = [["B-Prop", "I-Prop", "O"]]
        tags_b = [["B-Prop", "I-Prop", "O"]]
        result = merge_propositions([chunk_a, chunk_b], tags_a + tags_b, full)
        assert len(result) == 1
        assert result[0].start_char == 0
        assert result[0].end_char == 11
        assert sorted(result[0].chunk_offsets) == [0, 1]

    def test_longer_text_full_pipeline(self):
        sentence = "The system provides a robust framework for data processing. "
        text = sentence * 20
        chunks = chunk_text(text, tokenizer, max_tokens=50, overlap=10)
        assert len(chunks) >= 2
        all_tags = []
        for c in chunks:
            tags = ["O"] * len(c.tokens)
            if c.offset % 2 == 0 and len(tags) > 3:
                tags[2] = "B-Prop"
                for j in range(3, min(6, len(tags))):
                    tags[j] = "I-Prop"
            all_tags.append(tags)
        result = merge_propositions(chunks, all_tags, text)
        assert len(result) >= 1
        for span in result:
            assert span.end_char > span.start_char
            assert len(span.text) > 0
