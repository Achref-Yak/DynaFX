"""
fine_tune_roberta.py — Fine-tune DistilRoBERTa for Proposition Detection and Relation Classification.

Trains two models on UKP Argument Annotated Essays v2:
1. Token classifier (3 labels: O/B-Prop/I-Prop) — detects argumentative spans
2. Sequence classifier (3 labels: Support/Attack/None) — pairwise relation classification

Colab-compatible with mixed precision + gradient checkpointing to fit in 4GB VRAM (RTX 2050).

Key improvements over v1:
- Tagger: 20 epochs (was 3) → ~400 steps instead of ~60
- Classifier: weighted loss (Attack 10x, Support 3x) to fix class imbalance
- Classifier: undersamples "None" to 2x Support count for balanced batches
- Mixed precision (fp16) enabled by default for 2x speedup
- Cosine LR schedule with proportional warmup

Usage:
    python fine_tune_roberta.py                              # train both models
    python fine_tune_roberta.py --only-tagger                 # only train tagger
    python fine_tune_roberta.py --only-classifier              # only train classifier
    python fine_tune_roberta.py --colab-notebook              # generate Colab notebook
    python fine_tune_roberta.py --colab-notebook my.ipynb     # custom notebook path
    python fine_tune_roberta.py --data-dir ./data --output-dir ./models
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, classification_report
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TaggerDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }


class RelationDataset(Dataset):
    def __init__(self, pairs: list[dict], tokenizer, label_map: dict, max_length: int = 128):
        self.encodings = []
        self.labels = []
        for pair in pairs:
            enc = tokenizer(
                pair["span1"],
                pair["span2"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })
            self.labels.append(label_map[pair["label"]])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings[idx]["input_ids"],
            "attention_mask": self.encodings[idx]["attention_mask"],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def undersample_none(pairs: list[dict], ratio: float = 2.0) -> list[dict]:
    label_counts = {}
    for p in pairs:
        label_counts[p["label"]] = label_counts.get(p["label"], 0) + 1
    print(f"  Original distribution: {label_counts}")

    support_count = label_counts.get("Support", 1)
    none_target = int(support_count * ratio)

    kept = []
    none_seen = 0
    random.shuffle(pairs)
    for p in pairs:
        if p["label"] == "None":
            if none_seen < none_target:
                kept.append(p)
                none_seen += 1
        else:
            kept.append(p)

    new_counts = {}
    for p in kept:
        new_counts[p["label"]] = new_counts.get(p["label"], 0) + 1
    print(f"  After undersampling None (ratio={ratio}): {new_counts}")
    return kept


# ---------------------------------------------------------------------------
# Weighted Trainer (handles class imbalance for the relation classifier)
# ---------------------------------------------------------------------------

class WeightedClassifierTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tagger_compute_metrics(p):
    predictions = p.predictions.argmax(-1)
    labels = p.label_ids
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    accuracy = (predictions == labels).mean()

    prop_pred = (predictions > 0).astype(int)
    prop_true = (labels > 0).astype(int)
    prop_prec, prop_rec, prop_f1, _ = precision_recall_fscore_support(
        prop_true, prop_pred, average="binary", pos_label=1, zero_division=0
    )

    per_class = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], zero_division=0
    )
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "prop_f1": prop_f1,
        "prop_precision": prop_prec,
        "prop_recall": prop_rec,
        "O_f1": per_class[2][0],
        "B-Prop_f1": per_class[2][1] if len(per_class[2]) > 1 else 0,
        "I-Prop_f1": per_class[2][2] if len(per_class[2]) > 2 else 0,
    }


def classifier_compute_metrics(p):
    predictions = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    accuracy = (predictions == labels).mean()
    report = classification_report(
        labels, predictions, target_names=["None", "Support", "Attack"],
        zero_division=0, output_dict=True,
    )
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "attack_f1": report.get("Attack", {}).get("f1", 0),
        "attack_recall": report.get("Attack", {}).get("recall", 0),
        "support_f1": report.get("Support", {}).get("f1", 0),
        "support_recall": report.get("Support", {}).get("recall", 0),
    }


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def _make_training_args(
    output_dir: str,
    batch_size: int,
    grad_accum: int,
    lr: float,
    epochs: int,
    warmup_steps: int,
    metric_for_best: str = "f1",
    save_total: int = 2,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        bf16=not torch.cuda.is_available(),
        learning_rate=lr,
        num_train_epochs=epochs,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        report_to="none",
        save_total_limit=save_total,
        ddp_find_unused_parameters=False,
    )


def run_tagger_training(
    train_data: list[dict],
    test_data: list[dict],
    output_dir: str,
    model_name: str = "distilroberta-base",
    batch_size: int = 4,
    grad_accum: int = 4,
    epochs: int = 20,
    lr: float = 2e-5,
):
    print("\n" + "=" * 60)
    print("Training Proposition Detector (Token Classifier)")
    print(f"  Model: {model_name}")
    print(f"  Train docs: {len(train_data)}, Test docs: {len(test_data)}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Grad accum: {grad_accum}")
    print(f"  Effective batch: {batch_size * grad_accum}")
    print(f"  Steps per epoch: ~{len(train_data) // (batch_size * grad_accum)}")
    print("=" * 60)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "O", 1: "B-Prop", 2: "I-Prop"},
        label2id={"O": 0, "B-Prop": 1, "I-Prop": 2},
        ignore_mismatched_sizes=True,
    )

    train_dataset = TaggerDataset(train_data)
    eval_dataset = TaggerDataset(test_data)

    eff_batch = batch_size * grad_accum
    steps_per_epoch = max(1, len(train_data) // eff_batch)
    warmup_steps = max(10, steps_per_epoch // 10)

    training_args = _make_training_args(
        output_dir=output_dir,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        epochs=epochs,
        warmup_steps=warmup_steps,
        metric_for_best="prop_f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=tagger_compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    print(f"\nProposition Detector saved to {output_dir}")
    final_metrics = trainer.evaluate()
    print(f"  Final metrics: prop_f1={final_metrics.get('eval_prop_f1', 0):.4f}, "
          f"accuracy={final_metrics.get('eval_accuracy', 0):.4f}")


def run_classifier_training(
    train_pairs: list[dict],
    test_pairs: list[dict],
    label_map: dict,
    output_dir: str,
    model_name: str = "distilroberta-base",
    batch_size: int = 8,
    grad_accum: int = 4,
    epochs: int = 8,
    lr: float = 2e-5,
    max_length: int = 128,
):
    print("\n" + "=" * 60)
    print("Training Relation Classifier (Sequence Pair Classifier)")
    print(f"  Model: {model_name}")
    print(f"  Original train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Grad accum: {grad_accum}")
    print("=" * 60)

    train_pairs = undersample_none(train_pairs, ratio=2.0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = RelationDataset(train_pairs, tokenizer, label_map, max_length)
    eval_dataset = RelationDataset(test_pairs, tokenizer, label_map, max_length)

    eff_batch = batch_size * grad_accum
    steps_per_epoch = max(1, len(train_dataset) // eff_batch)
    warmup_steps = max(10, steps_per_epoch // 10)

    training_args = _make_training_args(
        output_dir=output_dir,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        epochs=epochs,
        warmup_steps=warmup_steps,
        metric_for_best="support_f1",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "None", 1: "Support", 2: "Attack"},
        label2id={"None": 0, "Support": 1, "Attack": 2},
        ignore_mismatched_sizes=True,
    )

    class_weights = torch.tensor([0.3, 3.0, 10.0], dtype=torch.float)

    trainer = WeightedClassifierTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=classifier_compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nRelation Classifier saved to {output_dir}")
    final_metrics = trainer.evaluate()
    print(f"  Final metrics: support_f1={final_metrics.get('eval_support_f1', 0):.4f}, "
          f"attack_f1={final_metrics.get('eval_attack_f1', 0):.4f}, "
          f"accuracy={final_metrics.get('eval_accuracy', 0):.4f}")


# ---------------------------------------------------------------------------
# Colab cell markers
# ---------------------------------------------------------------------------

def write_colab_notebook(output_path: str = "fine_tune_roberta.ipynb"):
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Fine-tune DistilRoBERTa for Cognitive Reasoning Engine\n",
                "\n",
                "Trains two models on UKP Argument Annotated Essays v2:\n",
                "1. **Proposition Detector** — token classifier (O/B-Prop/I-Prop)\n",
                "2. **Relation Classifier** — sequence pair classifier (Support/Attack/None)\n",
                "\n",
                "Run cells in order. Each cell is independent so you can resume from any step.",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Cell 1: Install dependencies\n",
                "!pip install torch transformers scikit-learn tqdm accelerate datasets\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Cell 2: Check GPU + VRAM\n",
                "import torch\n",
                "if torch.cuda.is_available():\n",
                "    gpu_name = torch.cuda.get_device_name()\n",
                "    vram = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
                '    print(f"GPU: {gpu_name}  |  VRAM: {vram:.1f} GB")\n',
                "    if vram < 4:\n",
                '        print("WARNING: <4GB VRAM. May OOM. Reduce batch sizes in Cell 5/6.")\n',
                "else:\n",
                '    print("ERROR: No GPU detected! Go to Runtime > Change runtime type > T4 GPU")\n',
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Cell 3: Mount Google Drive\n",
                "from google.colab import drive\n",
                'drive.mount("/content/drive")\n',
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                '# Cell 4: Prepare data (download UKP if not on Drive)\n',
                'import os, sys, json, zipfile, urllib.request\n',
                'import torch\n',
                'from collections import Counter\n',
                "\n",
                'DATA_DIR = "/content/drive/MyDrive/reasoning_engine/data"\n',
                'SCRIPT_DIR = "/content/drive/MyDrive/reasoning_engine/scripts"\n',
                'OUTPUT_DIR = "/content/drive/MyDrive/reasoning_engine/models"\n',
                "\n",
                "os.makedirs(DATA_DIR, exist_ok=True)\n",
                "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
                'sys.path.insert(0, SCRIPT_DIR)\n',
                "\n",
                'tagger_train = os.path.join(DATA_DIR, "ukp_tagger_train.pt")\n',
                "if not os.path.exists(tagger_train):\n",
                '    print("Downloading UKP Argument Annotated Essays v2...")\n',
                '    UKP_URL = ("https://tudatalib.ulb.tu-darmstadt.de/bitstreams/"\n',
                '               "1ae1718d-7e65-42ba-9e84-dbf52fe92f56/download")\n',
                '    zip_path = "/tmp/ukp.zip"\n',
                "    urllib.request.urlretrieve(UKP_URL, zip_path)\n",
                "    print(f\"  Downloaded {zip_path}\")\n",
                "    with zipfile.ZipFile(zip_path, \"r\") as z:\n",
                '        z.extractall("/tmp/ukp_extracted")\n',
                "    print(\"Processing into training format...\")\n",
                "    !python \"{SCRIPT_DIR}/prepare_ukp.py\" --data-dir \"{DATA_DIR}\"\n",
                '    print(f"Data saved to {DATA_DIR}")\n',
                "else:\n",
                '    print(f"Data already exists at {DATA_DIR}")\n',
                "\n",
                'train_t = torch.load(os.path.join(DATA_DIR, "ukp_tagger_train.pt"))\n',
                'test_t = torch.load(os.path.join(DATA_DIR, "ukp_tagger_test.pt"))\n',
                'with open(os.path.join(DATA_DIR, "ukp_relations_train.json")) as f:\n',
                "    train_r = json.load(f)\n",
                'print(f"Tagger: {len(train_t)} train / {len(test_t)} test docs")\n',
                'print(f"Relations: {len(train_r)} train pairs")\n',
                'print(f"  Distribution: {dict(Counter(p[\'label\'] for p in train_r))}")\n',
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                '# Cell 5: Train Proposition Detector (token classifier)\n',
                '# Expects ~20-30 min on T4 GPU\n',
                'print("=" * 60)\n',
                'print("TRAINING PROPOSITION DETECTOR")\n',
                'print("=" * 60)\n',
                '!python "{SCRIPT_DIR}/fine_tune_roberta.py" \\\n',
                '    --only-tagger \\\n',
                '    --data-dir "{DATA_DIR}" \\\n',
                '    --output-dir "{OUTPUT_DIR}" \\\n',
                "    --epochs 20\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                '# Cell 6: Train Relation Classifier (sequence pair classifier)\n',
                '# Expects ~10-15 min on T4 GPU\n',
                'print("=" * 60)\n',
                'print("TRAINING RELATION CLASSIFIER")\n',
                'print("=" * 60)\n',
                '!python "{SCRIPT_DIR}/fine_tune_roberta.py" \\\n',
                '    --only-classifier \\\n',
                '    --data-dir "{DATA_DIR}" \\\n',
                '    --output-dir "{OUTPUT_DIR}" \\\n',
                "    --epochs 8\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Cell 7: Verify models\n",
                "import os\n",
                "from transformers import (\n",
                "    AutoTokenizer,\n",
                "    AutoModelForTokenClassification,\n",
                "    AutoModelForSequenceClassification,\n",
                ")\n",
                "\n",
                'tag_dir = os.path.join(OUTPUT_DIR, "roberta-proposition-detector")\n',
                'cls_dir = os.path.join(OUTPUT_DIR, "roberta-relation-classifier")\n',
                "\n",
                "if os.path.exists(tag_dir):\n",
                "    model = AutoModelForTokenClassification.from_pretrained(tag_dir)\n",
                '    print(f"Proposition Detector: {sum(p.numel() for p in model.parameters()):,} params")\n',
                "if os.path.exists(cls_dir):\n",
                "    model2 = AutoModelForSequenceClassification.from_pretrained(cls_dir)\n",
                '    print(f"Relation Classifier:  {sum(p.numel() for p in model2.parameters()):,} params")\n',
                "\n",
                'print("\\nDone! Models saved to:")\n',
                'print(f"  {tag_dir}")\n',
                'print(f"  {cls_dir}")\n',
            ],
        },
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Colab notebook written to {os.path.abspath(output_path)}")
    print()
    print("Next steps:")
    print(f"  1. Upload the entire project folder to Google Drive at MyDrive/reasoning_engine/")
    print(f"  2. Upload {output_path} to Colab")
    print("  3. Run cells in order (Runtime > Run all)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilRoBERTa for proposition detection and relation classification"
    )
    parser.add_argument("--data-dir", default="./data", help="Directory with processed UKP data")
    parser.add_argument("--output-dir", default="./models", help="Directory to save trained models")
    parser.add_argument("--model-name", default="distilroberta-base", help="Base model name")
    parser.add_argument("--only-tagger", action="store_true", help="Only train the proposition detector")
    parser.add_argument("--only-classifier", action="store_true", help="Only train the relation classifier")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for the selected model(s)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--colab-notebook", type=str, nargs="?", const="fine_tune_roberta.ipynb", default=None, help="Generate a Colab .ipynb file (optional: specify output path)")
    args = parser.parse_args()

    if args.colab_notebook:
        write_colab_notebook(args.colab_notebook)
        return

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tagger_train_path = os.path.join(data_dir, "ukp_tagger_train.pt")
    tagger_test_path = os.path.join(data_dir, "ukp_tagger_test.pt")
    relations_train_path = os.path.join(data_dir, "ukp_relations_train.json")
    relations_test_path = os.path.join(data_dir, "ukp_relations_test.json")
    label_map_path = os.path.join(data_dir, "label_map.json")
    relation_map_path = os.path.join(data_dir, "relation_map.json")

    missing = []
    for p in [tagger_train_path, tagger_test_path, label_map_path]:
        if not os.path.exists(p):
            missing.append(p)
    if not args.only_tagger:
        for p in [relations_train_path, relations_test_path, relation_map_path]:
            if not os.path.exists(p):
                missing.append(p)
    if missing:
        print("Missing data files. Run prepare_ukp.py first:")
        print(f"  python prepare_ukp.py --data-dir {data_dir}")
        for p in missing:
            print(f"    missing: {p}")
        sys.exit(1)

    print("Loading data...")
    train_tagger = torch.load(tagger_train_path)
    test_tagger = torch.load(tagger_test_path)

    with open(label_map_path) as f:
        label_map = json.load(f)
    print(f"Label map: {label_map}")

    if not args.only_tagger:
        with open(relations_train_path) as f:
            train_relations = json.load(f)
        with open(relations_test_path) as f:
            test_relations = json.load(f)
        with open(relation_map_path) as f:
            relation_map = json.load(f)
        print(f"Relation map: {relation_map}")
        print(f"  Train relations: {len(train_relations)}")
        print(f"  Test relations:  {len(test_relations)}")

    print(f"  Train tagger: {len(train_tagger)} docs")
    print(f"  Test tagger:  {len(test_tagger)} docs")

    if not args.only_classifier:
        tag_epochs = args.epochs if args.epochs is not None else 20
        run_tagger_training(
            train_data=train_tagger,
            test_data=test_tagger,
            output_dir=os.path.join(output_dir, "roberta-proposition-detector"),
            model_name=args.model_name,
            epochs=tag_epochs,
            lr=args.lr,
        )

    if not args.only_tagger:
        cls_epochs = args.epochs if args.epochs is not None else 8
        run_classifier_training(
            train_pairs=train_relations,
            test_pairs=test_relations,
            label_map=relation_map,
            output_dir=os.path.join(output_dir, "roberta-relation-classifier"),
            model_name=args.model_name,
            epochs=cls_epochs,
            lr=args.lr,
        )

    print("\nDone! Models saved to:")
    print(f"  {os.path.join(output_dir, 'roberta-proposition-detector')}/")
    print(f"  {os.path.join(output_dir, 'roberta-relation-classifier')}/")


if __name__ == "__main__":
    main()
