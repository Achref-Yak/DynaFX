import argparse
import json
import logging
import sys
from pathlib import Path

from cognitive_engine.reason.evidence import CorpusResult
from cognitive_engine.reason.store import CorpusStore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to priors JSON config file (default: built-in defaults)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512,
        help="Maximum tokens per sliding window chunk (default: 512)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=128,
        help="Token overlap between adjacent chunks (default: 128)",
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if not path.exists():
        logger.error("File not found: %s", args.file)
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    logger.info("Processing file: %s (%d chars)", args.file, len(text))

    try:
        from cognitive_engine.pipeline.orchestrator import run as _orchestrator_run
        graph = _orchestrator_run(
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


def cmd_learn_priors(args: argparse.Namespace) -> None:
    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        logger.error("Corpus directory not found: %s", args.corpus)
        sys.exit(1)

    logger.info("Learning priors from corpus: %s", args.corpus)
    logger.info("Max files: %s", args.max_files or "unlimited")

    result = CorpusResult.from_corpus(
        corpus_dir,
        max_files=args.max_files,
        config_path=args.config,
    )

    if result.graph_count == 0:
        logger.error("No graphs produced — cannot learn priors")
        sys.exit(1)

    priors = result.to_priors()
    output = priors.to_dict()

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2))
        logger.info("Priors written to: %s", args.output)

    if args.save_corpus:
        result.save(args.save_corpus)

    logger.info("Learned from %d text(s)", result.graph_count)
    for key, op in priors.default_opinions.items():
        logger.info("  %s: (%.3f, %.3f, %.3f, %.3f)", key, *op)


def cmd_store_learn(args: argparse.Namespace) -> None:
    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        logger.error("Corpus directory not found: %s", args.corpus)
        sys.exit(1)

    store = CorpusStore(args.db)
    existing = {s["id"] for s in store.list_sources()}

    txt_files = sorted(corpus_dir.rglob("*.txt"))
    if args.max_files is not None:
        txt_files = txt_files[:args.max_files]

    if not txt_files:
        logger.warning("No .txt files found in %s", args.corpus)
        store.close()
        return

    new_count = 0
    for f in txt_files:
        sid = f.relative_to(corpus_dir).with_suffix("").as_posix()
        if sid in existing:
            logger.info("  Skipping %s (already stored)", f.name)
            continue

        try:
            from cognitive_engine.pipeline.orchestrator import run as _pipeline_run
            text = f.read_text(encoding="utf-8")
            graph = _pipeline_run(text, config_path=args.config)
            store.store_graph(graph, source_id=sid, filename=f.name)
            new_count += 1
            logger.info("  Stored %s (%d nodes, %d edges)", f.name, len(graph.nodes), len(graph.edges))
        except Exception as e:
            logger.warning("  Skipped %s: %s", f.name, e)

    store.close()
    logger.info("Stored %d new source(s) in %s", new_count, args.db)


def cmd_store_priors(args: argparse.Namespace) -> None:
    store = CorpusStore(args.db)
    if store.source_count == 0:
        logger.error("Store is empty — cannot export priors")
        store.close()
        sys.exit(1)

    priors = store.to_priors()
    output = priors.to_dict()
    Path(args.output).write_text(json.dumps(output, indent=2))
    logger.info("Priors written to: %s", args.output)
    store.close()


def cmd_store_list(args: argparse.Namespace) -> None:
    store = CorpusStore(args.db)
    sources = store.list_sources()
    store.close()

    if not sources:
        logger.info("Store is empty: %s", args.db)
        return

    logger.info("Sources in %s:", args.db)
    for s in sources:
        logger.info("  %s  |  %s  |  %s", s["id"], s["filename"], s["processed_at"])


def cmd_store_clear(args: argparse.Namespace) -> None:
    store = CorpusStore(args.db)
    count = store.source_count
    store.clear()
    store.close()
    logger.info("Cleared %d source(s) from %s", count, args.db)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cognitive Reasoning Graph Engine — CLI"
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand")

    analyze_p = sub.add_parser("analyze", help="Analyze a text file")
    analyze_p.add_argument("file", type=str, help="Path to text file to analyze")
    analyze_p.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: print to stdout)",
    )
    analyze_p.add_argument(
        "--mode", type=str, default=None,
        choices=["causal", "conditional", "argument", "analogy"],
        help="Reasoning mode to apply (default: all modes computed, argument returned)",
    )
    _add_common_args(analyze_p)
    analyze_p.set_defaults(func=cmd_analyze)

    learn_p = sub.add_parser("learn-priors", help="Learn priors from a corpus of .txt files")
    learn_p.add_argument("corpus", type=str, help="Directory containing .txt files")
    learn_p.add_argument(
        "--output", type=str, default=None,
        help="Output priors JSON file path",
    )
    learn_p.add_argument(
        "--save-corpus", type=str, default=None,
        help="Save corpus analysis result to file",
    )
    learn_p.add_argument(
        "--max-files", type=int, default=None,
        help="Maximum number of files to process from corpus",
    )
    _add_common_args(learn_p)
    learn_p.set_defaults(func=cmd_learn_priors)

    store_p = sub.add_parser("store", help="Persistent evidence store commands")
    store_sub = store_p.add_subparsers(dest="store_command", help="Store subcommand")

    store_learn_p = store_sub.add_parser("learn", help="Process a corpus into the store")
    store_learn_p.add_argument("corpus", type=str, help="Directory containing .txt files")
    store_learn_p.add_argument(
        "--db", type=str, default="evidence.db",
        help="SQLite database path (default: evidence.db)",
    )
    store_learn_p.add_argument(
        "--max-files", type=int, default=None,
        help="Maximum number of files to process",
    )
    _add_common_args(store_learn_p)
    store_learn_p.set_defaults(func=cmd_store_learn)

    store_priors_p = store_sub.add_parser("priors", help="Export store contents as priors JSON")
    store_priors_p.add_argument(
        "--db", type=str, default="evidence.db",
        help="SQLite database path (default: evidence.db)",
    )
    store_priors_p.add_argument(
        "--output", type=str, required=True,
        help="Output priors JSON file path",
    )
    store_priors_p.set_defaults(func=cmd_store_priors)

    store_list_p = store_sub.add_parser("list", help="List stored sources")
    store_list_p.add_argument(
        "--db", type=str, default="evidence.db",
        help="SQLite database path (default: evidence.db)",
    )
    store_list_p.set_defaults(func=cmd_store_list)

    store_clear_p = store_sub.add_parser("clear", help="Clear all stored data")
    store_clear_p.add_argument(
        "--db", type=str, default="evidence.db",
        help="SQLite database path (default: evidence.db)",
    )
    store_clear_p.set_defaults(func=cmd_store_clear)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "store" and args.store_command is None:
        store_p.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
