import csv
import logging
import math
import json
import os
import re
import time
from typing import List

from openai import OpenAI

from robomorph.evolution import llm
from robomorph.evolution.population import EvolutionUnit, Population
from robomorph.evolution.utils import (
    read_txt, write_txt, convert_dataset_to_zip,
    merge_datasets, design_score, prompts_path,
)
from compiler.compile import compile_robot, make_robot, xml_has_joints
from train.parallel_train import train_multiple_models

logger = logging.getLogger(__name__)


def create_population(size: int, problem_description: str) -> Population:
    """Create an initial population of robot designs."""
    return Population(
        size=size,
        problem_description=problem_description,
        elites=[],
        units=[EvolutionUnit() for _ in range(size)],
    )


def init_run(population: Population, exp_name: str, results_path: str):
    """Set up the problem description and create the results CSV file."""
    for unit in population.units:
        unit.prompt = population.problem_description

    if len(population.units) != population.size:
        raise ValueError(
            f"Population size mismatch: {len(population.units)} units but size={population.size}"
        )

    results_dir = os.path.dirname(results_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    with open(results_path, "x", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            ["generation", "unit_id", "xml_file", "structural", "component", "fitness", "diversity", "error"]
        )

    return population


def run_for_n(
    exp_name: str,
    num_generations: int,
    population: Population,
    client: OpenAI,
    terrain: str,
    model_name: str,
    results_path: str,
    seed: int = 42,
    timesteps: int = 1_000_000,
    top_k: int = 3,
    log: bool = True,
    evaluator: str = "sac",
    mpc_episodes: int = 3,
    mpc_threads: int = 8,
    domain: str = "rm",
    max_modules: int = 8,
    terrain_scale: float = 1.0,
):
    """Run the genetic algorithm for n generations."""
    results_dir = os.path.dirname(results_path)
    time_elapse_path = os.path.join(results_dir, f"time_elapse_{exp_name}.csv") if results_dir else f"time_elapse_{exp_name}.csv"
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    population_size = len(population.units)

    with open(time_elapse_path, "x", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["generation", "time_seconds"])
    usage_path = os.path.join(results_dir, f"usage_{exp_name}.csv") if results_dir else f"usage_{exp_name}.csv"
    with open(usage_path, "x", newline="") as f:
        csv.writer(f, delimiter=";").writerow(["generation", "llm_calls", "input_tokens", "output_tokens"])

    for generation in range(num_generations):
        start_time = time.time()
        logger.info(f"================== Generation {generation} ==================")

        _evaluate_fitness(
            exp_name, generation, population, client, terrain, model_name,
            results_path=results_path, seed=seed, timesteps=timesteps, top_k=top_k, log=log,
            evaluator=evaluator, mpc_episodes=mpc_episodes, mpc_threads=mpc_threads,
            domain=domain, max_modules=max_modules, terrain_scale=terrain_scale,
        )

        # Selection step: keep top designs
        population.units = select_top_units(population.units, population_size)

        elapsed = time.time() - start_time
        logger.info(f"Generation {generation} completed in {elapsed:.1f}s")

        with open(time_elapse_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([generation, f"{elapsed:.2f}"])
        with open(usage_path, "a", newline="") as f:  # cumulative totals
            csv.writer(f, delimiter=";").writerow(
                [generation, llm.USAGE["calls"], llm.USAGE["input_tokens"], llm.USAGE["output_tokens"]])

        with open(results_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            for unit in population.units:
                # Preserve the design's original ID after fitness sorting.
                xml_file = os.path.relpath(unit.xml_path, results_dir or ".") if unit.xml_path else ""
                writer.writerow([
                    generation,
                    unit.unit_id,
                    xml_file,
                    unit.structure,
                    unit.components,
                    unit.fitness,
                    unit.diversity,
                    unit.error,
                ])

        if not any(math.isfinite(unit.fitness) for unit in population.units):
            raise RuntimeError(f"Generation {generation} has no successful training results; inspect the results CSV and failure files")

    return population


def generate_grammar(client: OpenAI, system_prompt: str, user_prompt: str, model_name: str):
    """Prompt the LLM with the robot grammar and return the raw response."""
    if llm.provider() == "anthropic":
        content = llm.anthropic_chat(system_prompt, user_prompt, model_name)
    else:
        content = llm.openai_chat(client, system_prompt, user_prompt, model_name)
    if not content:
        raise ValueError("LLM returned an empty design response")
    return content


def extract_last_step(text):
    """Extract the structural graph from the last 'Step N:' line in LLM output."""
    step_pattern = re.compile(r"Step (\d+): (.+)")
    steps = step_pattern.findall(text)
    if not steps:
        return None

    max_step = max(steps, key=lambda x: int(x[0]))
    step_text = max_step[1]
    return step_text.split(")", 1)[1].strip() if ")" in step_text else step_text


def extract_bracket_line(text):
    """Extract the first line starting with '[' (the component graph) from LLM output."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("["):
            return stripped
    return None


def _evaluate_fitness(
    exp_name: str,
    generation: int,
    population: Population,
    client: OpenAI,
    terrain: str,
    model_name: str,
    results_path: str,
    seed: int = 42,
    timesteps: int = 1_000_000,
    top_k: int = 3,
    log: bool = True,
    evaluator: str = "sac",
    mpc_episodes: int = 3,
    mpc_threads: int = 8,
    domain: str = "rm",
    max_modules: int = 8,
    terrain_scale: float = 1.0,
) -> Population:
    """Evaluate fitness: LLM design -> compile -> train -> update population."""
    logger.info("Starting fitness evaluation...")
    start_time = time.time()

    results_dir = os.path.dirname(results_path) or "."

    elite_fitness = float("-inf")

    population_file = os.path.join(results_dir, f"population_{exp_name}.txt")
    curr_population = read_txt(population_file) if os.path.exists(population_file) else ""

    if domain == "hyblock":
        from hyblock import llm_io as hyblock_io
        grammar_system = hyblock_io.system_prompt(max_modules)
    elif domain == "rm":
        grammar_system = read_txt(prompts_path("grammar.txt"))
    else:
        raise ValueError(f"Unknown domain: {domain}")

    current_elite = None
    xml_paths = []
    unit_records = []

    xml_exp_dir = os.path.join(results_dir, f"examples_{exp_name}")
    os.makedirs(xml_exp_dir, exist_ok=True)
    if generation == 0:
        with open(os.path.join(xml_exp_dir, "grammar_prompt.txt"), "x") as f:
            f.write(grammar_system)

    for i, unit in enumerate(population.units):
        # Reset unit state for this generation's new design
        unit.unit_id = i
        unit.fitness = float("-inf")
        unit.error = None
        unit.xml_path = None

        grammar = generate_grammar(
            client, grammar_system,
            unit.prompt + " " + curr_population, model_name,
        )
        response_path = os.path.join(xml_exp_dir, f"evo{generation}_unit{i}.txt")
        with open(response_path, "x") as f:
            f.write(grammar)
        diversity = None
        if domain == "hyblock":
            structural_graph = hyblock_io.extract_rules(grammar)
            component_graph = hyblock_io.extract_design(grammar)
            if component_graph is not None:
                diversity = hyblock_io.diversity(component_graph)
                logger.info(f"unit{i}: {component_graph}")
                try:
                    robot_xml = hyblock_io.compile_text(component_graph, max_modules)
                    xml_path = os.path.join(xml_exp_dir, f"evo{generation}_unit{i}.xml")
                    with open(xml_path, "x") as f:
                        f.write(robot_xml)
                    xml_paths.append(xml_path)
                    unit.xml_path = xml_path
                except ValueError as e:     # hyblock.design.DesignError (kind in the message)
                    unit.error = f"Invalid design: {e}"
                    logger.warning("Invalid design for unit %d: %s", i, e)
            else:
                unit.error = "LLM response has no DESIGN line"
            unit.structure = structural_graph
            unit.components = component_graph
            unit.diversity = diversity
            continue

        structural_graph = extract_last_step(grammar)
        component_graph = extract_bracket_line(grammar)

        if component_graph is not None:
            diversity = design_score(structural_graph, component_graph)
            logger.info(f"unit{i}: {component_graph}")
            try:
                robot_xml = make_robot(compile_robot(component_graph))
                xml_path = os.path.join(xml_exp_dir, f"evo{generation}_unit{i}.xml")
                with open(xml_path, "x") as f:
                    f.write(robot_xml)
                if xml_has_joints(xml_path):
                    xml_paths.append(xml_path)
                    unit.xml_path = xml_path
                else:
                    unit.error = "Compiled design has no actuator joints"
            except Exception as e:
                unit.error = f"Compilation failed: {e}"
                logger.warning("Failed to compile unit %d: %s", i, e)

        else:
            unit.error = "LLM response has no component graph"

        unit.structure = structural_graph
        unit.components = component_graph
        unit.diversity = diversity

    if not xml_paths:
        raise RuntimeError(f"Generation {generation} produced no trainable designs; inspect the LLM responses and compiler warnings")

    # Evaluate the designs: SAC training (original RoboMorph) or CPU MPPI (mpc/)
    if evaluator == "sac":
        results = train_multiple_models(
            xml_paths,
            terrain=terrain,
            timesteps=timesteps,
            seed=seed,
            log=log,
            output_dir=os.path.join(results_dir, f"runs_{exp_name}"),
        )
    elif evaluator == "mpc":
        from mpc.evaluate import evaluate_multiple_designs
        results = evaluate_multiple_designs(
            xml_paths,
            terrain=terrain,
            seed=seed,
            episodes=mpc_episodes,
            threads=mpc_threads,
            output_dir=os.path.join(results_dir, f"runs_{exp_name}"),
            terrain_scale=terrain_scale,
        )
    else:
        raise ValueError(f"Unknown evaluator: {evaluator}")
    results_dict = dict(results)

    for unit in population.units:
        if unit.xml_path in results_dict:
            unit.fitness = results_dict[unit.xml_path]
            if not math.isfinite(unit.fitness):
                failure_path = os.path.join(results_dir, f"runs_{exp_name}", f"failure_{os.path.splitext(os.path.basename(unit.xml_path))[0]}.json")
                with open(failure_path) as f:
                    unit.error = json.load(f)["error"]

        if unit.fitness > elite_fitness:
            current_elite = unit.model_copy()
            elite_fitness = unit.fitness

        if math.isfinite(unit.fitness):
            unit_records.append((unit.structure, unit.components, unit.fitness))

    # Update population file with top designs
    old_records = convert_dataset_to_zip(curr_population)
    update_population = merge_datasets(old_records, unit_records, k=top_k)
    write_txt(population_file, update_population)

    if current_elite is not None:
        population.elites.append(current_elite)

    logger.info(f"Fitness evaluation completed in {time.time() - start_time:.1f}s")
    return population


def select_top_units(units: List[EvolutionUnit], population_size: int) -> List[EvolutionUnit]:
    """Select the top units by fitness to maintain population size."""
    return sorted(units, key=lambda u: u.fitness, reverse=True)[:population_size]
