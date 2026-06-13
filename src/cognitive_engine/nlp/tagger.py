from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification

from cognitive_engine.nlp.chunker import Chunk, PropSpan
from cognitive_engine.nlp.preprocessor import PreprocessedChunk

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_TAGGER_DIR = _MODELS_DIR / "roberta-proposition-detector"
_CLASSIFIER_DIR = _MODELS_DIR / "roberta-relation-classifier"

TAG_LABELS = ["O", "B-Prop", "I-Prop"]
REL_LABELS = ["None", "Support", "Attack"]


class PropositionTagger:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        path = model_path or str(_TAGGER_DIR)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def tag_chunk(self, chunk: Chunk) -> List[str]:
        enc = self.tokenizer(
            chunk.text,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**enc)
        preds = outputs.logits.argmax(dim=-1).squeeze(0).cpu().tolist()

        offsets = enc.encodings[0].offsets  # type: ignore
        labels: List[str] = []
        for token_idx, pred in enumerate(preds):
            start, end = offsets[token_idx]
            if start == 0 and end == 0:
                continue
            labels.append(TAG_LABELS[pred])
        return labels


class RelationClassifier:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        path = model_path or str(_CLASSIFIER_DIR)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def classify(self, text_a: str, text_b: str) -> str:
        enc = self.tokenizer(
            text_a, text_b,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**enc)
        pred = outputs.logits.argmax(dim=-1).item()
        return REL_LABELS[pred]


class SentenceTagger:
    def extract_spans(self, preprocessed: List[PreprocessedChunk], text: str) -> List[PropSpan]:
        spans: List[PropSpan] = []
        seen: set = set()
        for pp in preprocessed:
            doc = pp.doc
            chunk_start = doc.user_data.get("chunk_start_char", 0)
            for sent in doc.sents:
                start = chunk_start + sent.start_char
                end = chunk_start + sent.end_char
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)
                span_text = text[start:end]
                if not span_text.strip():
                    continue
                spans.append(PropSpan(
                    start_char=start,
                    end_char=end,
                    text=span_text,
                ))
        return spans
