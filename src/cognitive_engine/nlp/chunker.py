from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP = 128
DEFAULT_MERGE_MARGIN_CHARS = 20

TAG_O = "O"
TAG_B = "B-Prop"
TAG_I = "I-Prop"
PROP_TAGS = {TAG_B, TAG_I}


@dataclass
class Chunk:
    start_char: int
    end_char: int
    text: str
    tokens: List[int]
    offsets: List[Tuple[int, int]]
    offset: int


@dataclass
class PropSpan:
    start_char: int
    end_char: int
    text: str
    chunk_offsets: List[int] = field(default_factory=list)


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int = DEFAULT_OVERLAP,
) -> List[Chunk]:
    if not text:
        return []

    tokenizer.model_max_length = 1 << 30
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]

    if not input_ids:
        return []

    stride = max_tokens - overlap
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(input_ids):
        end = min(start + max_tokens, len(input_ids))
        chunk_ids = input_ids[start:end]
        chunk_offsets = offsets[start:end]
        chunk_char_start = offsets[start][0]
        last_valid = end - 1
        while last_valid >= start and offsets[last_valid][0] == offsets[last_valid][1]:
            last_valid -= 1
        if last_valid < start:
            start += stride
            continue
        chunk_char_end = offsets[last_valid][1]
        chunk_text_slice = text[chunk_char_start:chunk_char_end]
        chunks.append(Chunk(
            start_char=chunk_char_start,
            end_char=chunk_char_end,
            text=chunk_text_slice,
            tokens=chunk_ids,
            offsets=chunk_offsets,
            offset=chunk_idx,
        ))
        chunk_idx += 1
        if end == len(input_ids):
            break
        start += stride

    return chunks


def _resolve_tag(tag: str | int) -> str:
    if isinstance(tag, int):
        return {0: TAG_O, 1: TAG_B, 2: TAG_I}.get(tag, TAG_O)
    return tag


def merge_propositions(
    chunks: List[Chunk],
    tags_per_chunk: List[List[str | int]],
    source_text: str,
    merge_margin_chars: int = DEFAULT_MERGE_MARGIN_CHARS,
) -> List[PropSpan]:
    if not chunks or not tags_per_chunk:
        return []

    raw_spans: List[Tuple[int, int, List[int]]] = []

    for chunk_idx, chunk in enumerate(chunks):
        tags = tags_per_chunk[chunk_idx]
        if not chunk.offsets or not tags:
            continue
        min_len = min(len(tags), len(chunk.offsets))
        if min_len == 0:
            continue
        if len(tags) != len(chunk.offsets):
            logger.warning(
                "Chunk %d: tag count (%d) != offset count (%d), truncating",
                chunk_idx, len(tags), len(chunk.offsets),
            )
            tags = tags[:min_len]

        i = 0
        while i < len(tags):
            tag = _resolve_tag(tags[i])
            if tag == TAG_B:
                span_start = chunk.offsets[i][0]
                span_end = chunk.offsets[i][1]
                i += 1
                while i < len(tags) and _resolve_tag(tags[i]) == TAG_I:
                    span_end = chunk.offsets[i][1]
                    i += 1
                raw_spans.append((span_start, span_end, [chunk.offset]))
            elif tag == TAG_I:
                logger.debug(
                    "Chunk %d: bare I-Prop at token %d without B-Prop, treating as B-Prop",
                    chunk_idx, i,
                )
                span_start = chunk.offsets[i][0]
                span_end = chunk.offsets[i][1]
                i += 1
                while i < len(tags) and _resolve_tag(tags[i]) == TAG_I:
                    span_end = chunk.offsets[i][1]
                    i += 1
                raw_spans.append((span_start, span_end, [chunk.offset]))
            else:
                i += 1

    if not raw_spans:
        return []

    raw_spans.sort(key=lambda x: x[0])

    merged: List[Tuple[int, int, List[int]]] = [raw_spans[0]]
    for span in raw_spans[1:]:
        last = merged[-1]
        gap = span[0] - last[1]
        if gap <= merge_margin_chars:
            merged[-1] = (
                min(last[0], span[0]),
                max(last[1], span[1]),
                sorted(set(last[2] + span[2])),
            )
        else:
            merged.append(span)

    result = []
    for start, end, chunk_offsets in merged:
        result.append(PropSpan(
            start_char=start,
            end_char=end,
            text=source_text[start:end],
            chunk_offsets=chunk_offsets,
        ))

    return result
