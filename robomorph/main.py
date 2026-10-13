import argparse
import json
import logging
import os
import random
import re
import sys
from importlib.metadata import distributions
from pathlib import Path

from dotenv import load_dotenv


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def experiment_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("use letters, digits, underscores, dots or hyphens, starting with a letter or digit")
    return value


def build_parser():
    parser = argparse.ArgumentParser(
        description="RoboMorph: Evolve robot morphologies using LLMs."
    )
    parser.add_argument("-n", "--num_generations", default=100, type=positive_int)
    parser.add_argument("-p", "--problem", default=None, type=str)
    parser.add_argument(
        "--terrain", choices=["flat", "ridged", "beams", "frozen_lake"], default="flat"
    )
    parser.add_argument("--population_size", default=32, type=positive_int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--exp_name", type=experiment_name, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--results_path", type=str, default=None, help="Path to a new results CSV (relative to the working directory)")
    parser.add_argument("--timesteps", type=positive_int, default=1_000_000, help="Training timesteps per model")
    parser.add_argument("--top_k", type=positive_int, default=3, help="Top-k designs for few-shot examples")
    parser.add_argument("--no_log", action="store_true", help="Disable Weights & Biases logging")
    return parser


def run_artifacts(results_path, exp_name):
    """Paths owned by one evolution run, checked before writing anything."""
    directory = Path(results_path).parent
    return [
        Path(results_path),
        directory / f"arguments_{exp_name}.json",
        directory / f"environment_{exp_name}.json",
        directory / f"time_elapse_{exp_name}.csv",
        directory / f"population_{exp_name}.txt",
        directory / f"examples_{exp_name}",
        directory / f"runs_{exp_name}",
    ]


def main():
    parser = build_parser()
    args = parser.parse_args()
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    if not os.environ.get("OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"] == "your-openai-api-key-here":
        parser.error("set OPENAI_API_KEY in the environment or .env before starting evolution")

    if args.results_path is None:
        args.results_path = str(Path("results") / f"results_{args.exp_name}.csv")
    existing = [str(path) for path in run_artifacts(args.results_path, args.exp_name) if path.exists()]
    if existing:
        parser.error("run artifacts already exist; choose a new --exp_name or output directory: " + ", ".join(existing))

    # Keep help, validation and compiler-only usage independent of the RL stack.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import numpy as np
    from openai import OpenAI
    from robomorph.evolution import create_population, init_run, run_for_n
    from robomorph.evolution.utils import read_txt, prompts_path
    from train.parallel_train import gpu_ids

    gpu_ids()  # Fail before consuming LLM tokens when GPU training is unavailable.
    if args.problem is None:
        args.problem = read_txt(prompts_path("user.txt"))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    client = OpenAI()
    random.seed(args.seed)
    np.random.seed(args.seed)

    results_dir = Path(args.results_path).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / f"arguments_{args.exp_name}.json").open("x") as f:
        json.dump(vars(args), f, indent=4)
    with (results_dir / f"environment_{args.exp_name}.json").open("x") as f:
        json.dump({"python": sys.version, "packages": {
            dist.metadata["Name"]: dist.version for dist in distributions()
        }}, f, indent=2, sort_keys=True)

    logger.info("Evolving robot designs for: %s", args.problem)
    population = create_population(args.population_size, args.problem)
    init_run(population, args.exp_name, args.results_path)
    population = run_for_n(
        exp_name=args.exp_name,
        num_generations=args.num_generations,
        population=population,
        client=client,
        terrain=args.terrain,
        model_name=args.model,
        results_path=args.results_path,
        seed=args.seed,
        timesteps=args.timesteps,
        top_k=args.top_k,
        log=not args.no_log,
    )

    best_unit = max(population.units, key=lambda u: u.fitness)
    print("=" * 80)
    print(f"Done! Best fitness: {best_unit.fitness}")
    print(f"Best design: {best_unit.components}")


if __name__ == "__main__":
    main()
