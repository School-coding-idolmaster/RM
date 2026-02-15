import argparse
import json
import logging
import os
import random

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rich import print
from multiprocessing import set_start_method

from robomorph.evolution import create_population, init_run, run_for_n
from robomorph.evolution.utils import read_txt, prompts_path

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    set_start_method("spawn", force=False)
    load_dotenv()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    user_prompt_path = prompts_path("user.txt")
    default_problem = read_txt(user_prompt_path) if os.path.exists(user_prompt_path) else "Design a robot."

    parser = argparse.ArgumentParser(
        description="RoboMorph: Evolve robot morphologies using LLMs."
    )
    parser.add_argument("-n", "--num_generations", default=100, type=int)
    parser.add_argument("-p", "--problem", default=default_problem, type=str)
    parser.add_argument(
        "--terrain", choices=["flat", "ridged", "beams", "frozen_lake"], default="flat"
    )
    parser.add_argument("--population_size", default=32, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--results_path", type=str, default=None, help="Path to save results CSV")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Training timesteps per model")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k designs for few-shot examples")

    args = parser.parse_args()

    if args.results_path is None:
        args.results_path = os.path.join(_PROJECT_ROOT, "results", f"results_{args.exp_name}.csv")

    results_dir = os.path.dirname(args.results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir or ".", f"arguments_{args.exp_name}.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    random.seed(args.seed)
    np.random.seed(args.seed)

    client = OpenAI()

    logger.info(f"Evolving robot designs for: {args.problem}")

    logger.info("Creating the population...")
    population = create_population(
        size=args.population_size, problem_description=args.problem
    )

    logger.info("Generating the initial prompts...")
    init_run(population, args.exp_name, args.results_path)

    logger.info("Starting the genetic algorithm...")
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
    )

    best_unit = max(population.units, key=lambda u: u.fitness)
    print("=" * 80)
    print(f"Done! Best fitness: {best_unit.fitness}")
    print(f"Best design: {best_unit.components}")


if __name__ == "__main__":
    main()
