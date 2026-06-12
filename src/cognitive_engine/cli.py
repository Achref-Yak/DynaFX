import argparse
import logging
import sys
from pathlib import Path

from cognitive_engine.orchestrator import run as orchestrator_run

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cognitive Reasoning Graph Engine — CLI"
    )
    parser.add_argument("file", type=str, help="Path to text file to analyze")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: print to stdout)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to priors JSON config file (default: built-in defaults)",
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["causal", "conditional", "argument", "analogy"],
        help="Reasoning mode to apply (default: all modes computed, argument returned)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512,
        help="Maximum tokens per sliding window chunk (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=128,
        help="Token overlap between adjacent chunks (default: 128)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        logger.error("File not found: %s", args.file)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    logger.info("Processing file: %s (%d chars)", args.file, len(text))

    try:
        graph = orchestrator_run(
            text,
            config_path=args.config,
            mode=args.mode,
            max_tokens=args.chunk_size,
            overlap=args.chunk_overlap,
        )
        result = graph.to_json()
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(result)
        logger.info("Output written to: %s", args.output)
    else:
        print(result)


if __name__ == "__main__":
    main()
