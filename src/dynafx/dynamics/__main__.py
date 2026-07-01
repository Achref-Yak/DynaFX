"""CLI entry point for .sysd model simulation and validation.

Usage:
    python -m dynafx.dynamics simulate <file.sysd> [options]
    python -m dynafx.dynamics validate <file.sysd>
    python -m dynafx.dynamics list
"""

import argparse
import csv
import copy
import glob
import os
import sys
from pathlib import Path

from dynafx.dynamics.dsl import parse_sysd_file, parse_sysd


def _find_library_models() -> list[tuple[str, str]]:
    models_dir = Path(__file__).parent.parent.parent.parent / "models"
    results: list[tuple[str, str]] = []
    if models_dir.is_dir():
        for fpath in sorted(models_dir.glob("*.sysd")):
            try:
                content = fpath.read_text(encoding="utf-8")
                first_line = content.strip().split("\n")[0][:80] if content else ""
                results.append((fpath.stem, first_line))
            except Exception:
                results.append((fpath.stem, ""))
    return results


def cmd_simulate(args: argparse.Namespace) -> None:
    model = parse_sysd_file(args.file)
    if args.dt is not None:
        model.dt = args.dt
    if args.t_end is not None:
        t0 = model.t_span[0]
        model.t_span = (t0, args.t_end)

    params = {}
    if args.param:
        for p in args.param:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k.strip()] = float(v.strip())

    # Filter paradigms if --paradigm is set
    paradigm = args.paradigm
    if paradigm != "all":
        model = copy.deepcopy(model)
    if paradigm == "abm":
        model.stocks = []
        model.flows = []
        model.aux_vars = []
        model.tables = []
        model.queues = []
        model.resources = []
        model.events = []
    elif paradigm == "des":
        model.stocks = []
        model.flows = []
        model.aux_vars = []
        model.tables = []
        model.agents = []
    elif paradigm == "sd":
        model.agents = []
        model.queues = []
        model.resources = []
        model.events = []

    result = model.simulate(method=args.method, params=params)

    print(f"Model: {model.name}")
    print(f"Method: {args.method}, Steps: {result.steps}")

    if result.stocks:
        print(f"Final state:")
        for i, name in enumerate(result.stocks):
            print(f"  {name}: {result.final_state[i]:.4f}")

    if result.abm_engine:
        print(f"\nABM: {len(result.abm_engine.instances)} agents")
        for inst in result.abm_engine.instances[:5]:
            print(f"  {inst.agent_def.name}[{inst.id}]: {inst.state}")

    if result.des_engine:
        print(f"\nDES Statistics:")
        for name, stats in result.des_engine.get_all_stats().items():
            print(f"  {name}: {stats}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t"] + result.stocks)
            for ti, t in enumerate(result.times):
                row = [f"{t:.6f}"] + [f"{result.values[s][ti]:.6f}" for s in result.stocks]
                writer.writerow(row)
        print(f"\nCSV written to {args.csv}")

    if args.output:
        result.plot(args.output)
        print(f"Plot saved to {args.output}")


def cmd_validate(args: argparse.Namespace) -> None:
    model = parse_sysd_file(args.file)
    print(f"Validating: {model.name}")
    result = model.validate()
    result.print_report()
    if not result.is_valid:
        sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    models = _find_library_models()
    if not models:
        print("No .sysd models found in models/ directory.")
        return
    print(f"{'Model':<30} Description")
    print("-" * 70)
    for name, desc in models:
        desc_short = desc[:60] if desc else "(no description)"
        print(f"{name:<30} {desc_short}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="System Dynamics DSL — simulate, validate, and explore .sysd models"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="Simulate a .sysd model")
    sim.add_argument("file", type=str, help="Path to .sysd model file")
    sim.add_argument("--method", choices=["rk4", "euler"], default="rk4")
    sim.add_argument("--dt", type=float, default=None, help="Override time step")
    sim.add_argument("--t-end", type=float, default=None, help="Override end time")
    sim.add_argument("--output", type=str, default=None, help="Save plot to file")
    sim.add_argument("--csv", type=str, default=None, help="Save trajectory to CSV")
    sim.add_argument("--param", type=str, action="append", default=None,
                     help="Set parameter: key=value (can repeat)")
    sim.add_argument("--paradigm", choices=["sd", "abm", "des", "all"], default="all",
                     help="Which paradigm to simulate (default: all)")

    val = sub.add_parser("validate", help="Validate a .sysd model")
    val.add_argument("file", type=str, help="Path to .sysd model file")

    lst = sub.add_parser("list", help="List available library models")

    args = parser.parse_args()

    if args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
