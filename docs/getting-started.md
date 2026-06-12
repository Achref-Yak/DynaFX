# Getting Started

## Requirements

- Python 3.12+
- A CUDA-capable GPU (optional, for RoBERTa models; falls back to CPU)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd reasoning_engine

# Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .

# Download the spaCy model (required for preprocessing)
python -m spacy download en_core_web_trf
```

## Quick Start

Run the pipeline on the included demo file:

```bash
cognitive-engine demo/complicated.txt
```

This prints a JSON graph to stdout. To save to a file:

```bash
cognitive-engine demo/complicated.txt --output result.json
```

## Understanding the Output

The output is a `Graph` object serialized as JSON with four top-level keys:

| Key | Description |
|-----|-------------|
| `nodes` | Map of UUID → typed node, each with text, span, opinion, category, and demarcation metadata |
| `edges` | List of typed edges with source/target node references |
| `mode` | Active reasoning mode (default: `ARGUMENT`) |
| `metadata` | Priors configuration and multi-mode view counts |
| `cta` | Conversation Tree Architecture — parent-child hierarchy rooted at the first unattacked node |

Each node has an `opinion` tuple `(belief, disbelief, uncertainty, base_rate)` satisfying `b + d + u = 1`.

## Running on Your Own Text

```bash
cognitive-engine path/to/your-file.txt
```

The input should be plain text (`.txt`). The pipeline handles documents of any length by splitting into overlapping token windows.

## Next Steps

- [Architecture](architecture.md) — understand how the pipeline fits together
- [CLI Reference](cli.md) — all available flags and options
- [Development](development.md) — running tests, training models
