import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from cognitive_engine.orchestrator import run, CafGenError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def run_pipeline(
    text: str,
    model: str | None = None,
    config: str | None = None,
    sweep: bool = False,
) -> str:
    graph = await run(
        text=text,
        model_name=model,
        best_effort=True,
        config_path=config,
        sweep=sweep,
    )
    return graph.to_json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cognitive Reasoning Graph Engine — CLI"
    )
    parser.add_argument("file", type=str, help="Path to text file to analyze")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Groq model name (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: print to stdout)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to priors JSON config file (default: built-in defaults)",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run sensitivity sweep: vary each prior ±0.1 and report opinion ranges",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        logger.error("File not found: %s", args.file)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    logger.info("Processing file: %s (%d chars)", args.file, len(text))

    try:
        result = asyncio.run(
            run_pipeline(text, args.model, args.config, args.sweep)
        )
    except CafGenError as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(result)
        logger.info("Output written to: %s", args.output)
    else:
        print(result)


if __name__ == "__main__":
    main()
