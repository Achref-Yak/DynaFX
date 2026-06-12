"""
prepare_ukp.py — Download UKP Argument Annotated Essays v2 and convert to binary Prop/O format.

Downloads the raw BRAT-format zip from TU Darmstadt (no PIE/HuggingFace dependencies),
parses .ann + .txt files, converts to token-level binary IOB labels, extracts relation
pairs, tokenizes with DistilRoBERTa tokenizer, and saves train/test splits to data/.

Usage:
    python prepare_ukp.py                          # full pipeline
    python prepare_ukp.py --zip path/to/local.zip  # skip download, use local zip
    python prepare_ukp.py --data-dir ./data         # custom output directory
"""

import argparse
import json
import os
import sys
import urllib.request
import zipfile

# Load .env (maps HG_TOKEN -> HF_TOKEN for HuggingFace)
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k == "HG_TOKEN":
                    os.environ["HF_TOKEN"] = _v
                elif not _k.startswith("LLM_") and _k not in os.environ:
                    os.environ[_k] = _v

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

UKP_URL = (
    "https://tudatalib.ulb.tu-darmstadt.de/bitstreams/"
    "1ae1718d-7e65-42ba-9e84-dbf52fe92f56/download"
)
ZIP_FILENAME = "ArgumentAnnotatedEssays-2.0.zip"


def download_ukp(target_dir: str) -> str:
    """Download the UKP v2 zip from TU Darmstadt. Returns path to zip file."""
    zip_path = os.path.join(target_dir, ZIP_FILENAME)
    if os.path.exists(zip_path):
        print(f"Zip already exists at {zip_path}, skipping download")
        return zip_path

    print(f"Downloading UKP v2 from {UKP_URL} ...")
    urllib.request.urlretrieve(UKP_URL, zip_path)
    print(f"Downloaded {zip_path}")
    return zip_path


def extract_zip(zip_path: str, extract_dir: str) -> tuple[str, str]:
    """Extract zip and return (inner_brat_zip_path, train_test_csv_path)."""
    print(f"Extracting to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    brat_zip = None
    csv_path = None
    for root, _dirs, files in os.walk(extract_dir):
        for f in files:
            if f == "brat-project-final.zip":
                brat_zip = os.path.join(root, f)
            elif f == "train-test-split.csv":
                csv_path = os.path.join(root, f)
    return brat_zip, csv_path


def extract_and_load(
    brat_zip_path: str,
) -> tuple[dict[str, str], dict[str, tuple]]:
    """Extract the inner brat-project-final.zip and return all texts + annotations.

    Returns:
        texts: dict[essay_id] -> raw text string
        annotations: dict[essay_id] -> (texts_dict, relations_list)
    """
    with zipfile.ZipFile(brat_zip_path, "r") as z:
        prefix = "brat-project-final/"
        texts = {}
        annotations = {}

        # Collect all .txt and .ann files (only from brat-project-final/, not __MACOSX/)
        txt_files = {
            n for n in z.namelist()
            if n.startswith(prefix) and n.endswith(".txt") and "/." not in n
        }
        ann_files = {
            n for n in z.namelist()
            if n.startswith(prefix) and n.endswith(".ann") and "/." not in n
        }

        for name in sorted(txt_files):
            essay_id = os.path.basename(name).replace(".txt", "")
            texts[essay_id] = z.read(name).decode("utf-8")

        for name in sorted(ann_files):
            essay_id = os.path.basename(name).replace(".ann", "")
            content = z.read(name).decode("utf-8")
            ann_texts, relations = _parse_ann_content(content)
            annotations[essay_id] = (ann_texts, relations)

        return texts, annotations


def _parse_ann_content(content: str) -> tuple[dict, list]:
    """Parse BRAT .ann file content string."""
    texts = {}
    relations = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("T"):
            parts = line.split("\t")
            tid = parts[0]
            label, start, end = parts[1].split()
            text = parts[2] if len(parts) > 2 else ""
            texts[tid] = (label, int(start), int(end), text)
        elif line.startswith("R"):
            parts = line.split()
            rtype = parts[1]
            arg1 = parts[2].split(":")[1]
            arg2 = parts[3].split(":")[1]
            relations.append((rtype, arg1, arg2))
    return texts, relations
    """Parse a BRAT .ann file.

    Returns:
        texts: dict[T_id] -> (label, start_char, end_char, text)
        relations: list[(type, arg1_id, arg2_id)]
    """
    texts = {}
    relations = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("T"):
                parts = line.split("\t")
                tid = parts[0]
                label, start, end = parts[1].split()
                text = parts[2] if len(parts) > 2 else ""
                texts[tid] = (label, int(start), int(end), text)

            elif line.startswith("R"):
                parts = line.split()
                rtype = parts[1]
                arg1 = parts[2].split(":")[1]
                arg2 = parts[3].split(":")[1]
                relations.append((rtype, arg1, arg2))

    return texts, relations


def make_tagger_labels(
    text: str, texts: dict, tokenizer, max_length: int = 512
) -> list[int]:
    """Convert BRAT annotations to token-level IOB labels.

    Returns list of ints: 0=O, 1=B-Prop, 2=I-Prop.
    Length = number of subword tokens (padded to max_length).
    """
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    offsets = encoding["offset_mapping"]
    labels = [0] * len(offsets)

    prop_spans = [
        (s, e)
        for t_id, (label, s, e, _) in texts.items()
        if label in ("MajorClaim", "Claim", "Premise")
    ]

    for span_start, span_end in prop_spans:
        first = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start is None or tok_end is None:
                continue
            if tok_start >= span_start and tok_end <= span_end:
                if first:
                    labels[i] = 1  # B-Prop
                    first = False
                else:
                    labels[i] = 2  # I-Prop

    return labels


def make_relation_data(
    text: str, texts: dict, relations: list, window_size: int = 5
) -> list[dict]:
    """Extract Support/Attack/None pairs between propositions.

    Only generates None pairs for propositions within `window_size` of each other
    (avoids O(n^2) explosion of negative samples).
    """
    prop_spans = [
        (tid, s, e, t)
        for tid, (label, s, e, t) in texts.items()
        if label in ("MajorClaim", "Claim", "Premise")
    ]

    # Build lookup: relation type by (arg1, arg2)
    rel_map = {}
    for rtype, a1, a2 in relations:
        rel_map[(a1, a2)] = rtype
        rel_map[(a2, a1)] = rtype

    result = []
    for i, (tid1, s1, e1, t1) in enumerate(prop_spans):
        for j, (tid2, s2, e2, t2) in enumerate(prop_spans):
            if i == j:
                continue

            rel = rel_map.get((tid1, tid2))
            if rel:
                label = "Support" if rel.lower() == "supports" else "Attack"
            else:
                if abs(i - j) > window_size:
                    continue
                label = "None"

            result.append({
                "span1": text[s1:e1],
                "span2": text[s2:e2],
                "label": label,
            })

    return result


def process_essays(
    essay_ids: list[str],
    texts: dict[str, str],
    annotations: dict[str, tuple],
    tokenizer,
    max_length: int = 512,
    window_size: int = 5,
    split_name: str = "",
) -> tuple[list[dict], list[dict]]:
    """Process a list of essay IDs into tagger and relation data.

    Returns:
        tagger_data: list of dicts with input_ids, attention_mask, labels
        relation_data: list of dicts with span1, span2, label
    """
    tagger_data = []
    relation_data = []

    for essay_id in tqdm(essay_ids, desc=f"Processing {split_name}"):
        if essay_id not in texts or essay_id not in annotations:
            print(f"Warning: {essay_id} missing text or annotations, skipping")
            continue

        text = texts[essay_id]
        ann_texts, ann_relations = annotations[essay_id]

        # Tagger data
        labels = make_tagger_labels(text, ann_texts, tokenizer, max_length)
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        tagger_data.append({
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        })

        # Relation data
        rels = make_relation_data(text, ann_texts, ann_relations, window_size)
        relation_data.extend(rels)

    return tagger_data, relation_data


def read_split_csv(csv_path: str) -> tuple[list[str], list[str]]:
    """Read train-test-split.csv and return (train_ids, test_ids)."""
    train_ids = []
    test_ids = []
    with open(csv_path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 2:
                continue
            essay_id = parts[0].strip('"')
            split_set = parts[1].strip('"')
            if split_set == "TRAIN":
                train_ids.append(essay_id)
            elif split_set == "TEST":
                test_ids.append(essay_id)
    return train_ids, test_ids


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare UKP Argument Annotated Essays v2"
    )
    parser.add_argument(
        "--data-dir", default="./data", help="Output directory for processed data"
    )
    parser.add_argument(
        "--zip", default=None, help="Path to local zip file (skip download)"
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Max tokenizer length"
    )
    parser.add_argument(
        "--window-size", type=int, default=5, help="Max span distance for negative pairs"
    )
    parser.add_argument(
        "--model-name",
        default="distilroberta-base",
        help="HuggingFace model name for tokenizer",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # Download
    if args.zip:
        zip_path = args.zip
        print(f"Using local zip: {zip_path}")
    else:
        zip_path = download_ukp(data_dir)

    # Extract outer zip to find inner brat zip + split CSV
    extract_dir = os.path.join(data_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    brat_zip_path, csv_path = extract_zip(zip_path, extract_dir)

    if not brat_zip_path:
        print("Error: could not find brat-project-final.zip inside archive")
        sys.exit(1)
    if not csv_path:
        print("Error: could not find train-test-split.csv inside archive")
        sys.exit(1)

    print(f"BRAT zip: {brat_zip_path}")
    print(f"Split CSV: {csv_path}")

    # Load all texts + annotations from the inner brat zip
    print("Loading BRAT annotations...")
    texts, annotations = extract_and_load(brat_zip_path)
    print(f"  Loaded {len(texts)} texts, {len(annotations)} annotation sets")

    # Read train/test split
    train_ids, test_ids = read_split_csv(csv_path)
    print(f"  Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Process splits
    train_tagger, train_relations = process_essays(
        train_ids, texts, annotations, tokenizer, args.max_length, args.window_size,
        split_name="TRAIN",
    )
    test_tagger, test_relations = process_essays(
        test_ids, texts, annotations, tokenizer, args.max_length, args.window_size,
        split_name="TEST",
    )

    # Save tagger data
    print("Saving tagger data...")
    torch.save(train_tagger, os.path.join(data_dir, "ukp_tagger_train.pt"))
    torch.save(test_tagger, os.path.join(data_dir, "ukp_tagger_test.pt"))

    # Save relation data
    print("Saving relation data...")
    with open(os.path.join(data_dir, "ukp_relations_train.json"), "w") as f:
        json.dump(train_relations, f)
    with open(os.path.join(data_dir, "ukp_relations_test.json"), "w") as f:
        json.dump(test_relations, f)

    # Save label maps
    label_map = {"O": 0, "B-Prop": 1, "I-Prop": 2}
    relation_map = {"None": 0, "Support": 1, "Attack": 2}
    with open(os.path.join(data_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)
    with open(os.path.join(data_dir, "relation_map.json"), "w") as f:
        json.dump(relation_map, f)

    # Stats
    print(f"\nDone! Stats:")
    print(f"  Train tagger: {len(train_tagger)} docs")
    print(f"  Test tagger:  {len(test_tagger)} docs")
    print(f"  Train relations: {len(train_relations)} pairs")
    print(f"  Test relations:  {len(test_relations)} pairs")

    # Relation label distribution
    for split_name, rels in [("Train", train_relations), ("Test", test_relations)]:
        counts = {}
        for r in rels:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        print(f"  {split_name} relation distribution: {counts}")


if __name__ == "__main__":
    main()
