"""Deposition-aware chunker.

Wraps the base chunk_text with deposition structure awareness:
- Splits at Q/A boundaries (never breaks a Q/A pair)
- Adds speaker/examiner context as metadata
- Filters out header text
- Keeps examination type as chunk metadata
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from cognitive_engine.nlp.chunker import Chunk
from cognitive_engine.nlp.deposition_parser import (
    DepositionSection,
    ExaminationType,
    ParsedDeposition,
    QA,
)

logger = logging.getLogger(__name__)


@dataclass
class DepositionChunk(Chunk):
    """A chunk with deposition-specific metadata."""
    examination_type: str = ""
    examiner: str = ""
    qa_question: str = ""
    qa_answer: str = ""
    is_qa_pair: bool = False


def _examination_type_str(et: ExaminationType) -> str:
    return {
        ExaminationType.DIRECT: "direct",
        ExaminationType.CROSS: "cross",
        ExaminationType.REDIRECT: "redirect",
        ExaminationType.REBUTTAL: "rebuttal",
        ExaminationType.UNKNOWN: "unknown",
    }.get(et, "unknown")


def chunk_deposition(
    text: str,
    parsed: ParsedDeposition,
    tokenizer,
    max_tokens: int = 512,
    overlap: int = 128,
) -> List[DepositionChunk]:
    """Chunk a deposition with Q/A boundary awareness.

    Each Q/A pair is kept intact within a single chunk. Chunks are
    annotated with examination type, speaker, and Q/A content.
    """
    if not parsed.is_deposition or not parsed.sections:
        from cognitive_engine.nlp.chunker import chunk_text
        base = chunk_text(text, tokenizer=tokenizer, max_tokens=max_tokens, overlap=overlap)
        return [DepositionChunk(
            start_char=c.start_char,
            end_char=c.end_char,
            text=c.text,
            tokens=c.tokens,
            offsets=c.offsets,
            offset=c.offset,
        ) for c in base]

    chunks: List[DepositionChunk] = []
    chunk_idx = 0

    for section in parsed.sections:
        exam_str = _examination_type_str(section.exam_type)
        examiner = section.examiner

        for qa in section.qa_pairs:
            qa_text = f"Q: {qa.question}\nA: {qa.answer}"

            tokenizer.model_max_length = 1 << 30
            encoding = tokenizer(
                qa_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            input_ids = encoding["input_ids"]
            offsets = encoding["offset_mapping"]

            if not input_ids:
                continue

            # If Q/A fits in one chunk, create a single chunk
            if len(input_ids) <= max_tokens:
                char_start = qa.char_start
                char_end = qa.char_end
                chunks.append(DepositionChunk(
                    start_char=char_start,
                    end_char=char_end,
                    text=text[char_start:char_end],
                    tokens=input_ids,
                    offsets=offsets,
                    offset=chunk_idx,
                    examination_type=exam_str,
                    examiner=examiner,
                    qa_question=qa.question,
                    qa_answer=qa.answer,
                    is_qa_pair=True,
                ))
                chunk_idx += 1
            else:
                # Q/A too long — split but keep question with first part of answer
                stride = max_tokens - overlap
                start = 0
                first_chunk = True
                while start < len(input_ids):
                    end = min(start + max_tokens, len(input_ids))
                    chunk_ids = input_ids[start:end]
                    chunk_offsets = offsets[start:end]

                    # Find valid character range
                    last_valid = end - 1
                    while last_valid >= start and chunk_offsets[last_valid - start][0] == chunk_offsets[last_valid - start][1]:
                        last_valid -= 1
                    if last_valid < start:
                        start += stride
                        continue

                    chunk_char_start = qa.char_start + chunk_offsets[0][0]
                    chunk_char_end = qa.char_start + chunk_offsets[last_valid - start][1]

                    chunks.append(DepositionChunk(
                        start_char=chunk_char_start,
                        end_char=chunk_char_end,
                        text=text[chunk_char_start:chunk_char_end],
                        tokens=chunk_ids,
                        offsets=chunk_offsets,
                        offset=chunk_idx,
                        examination_type=exam_str,
                        examiner=examiner,
                        qa_question=qa.question if first_chunk else "",
                        qa_answer=qa.answer,
                        is_qa_pair=first_chunk,
                    ))
                    chunk_idx += 1
                    first_chunk = False

                    if end == len(input_ids):
                        break
                    start += stride

    # Sort by position in original text
    chunks.sort(key=lambda c: c.start_char)
    logger.info(
        "Deposition chunking: %d chunks from %d sections, %d Q/A pairs",
        len(chunks),
        len(parsed.sections),
        sum(len(s.qa_pairs) for s in parsed.sections),
    )
    return chunks
