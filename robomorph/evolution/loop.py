import csv
import logging
import os
import re
import time
from typing import List

from openai import OpenAI

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
    try:
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(
                ["generation", "unit_id", "xml_file", "structural", "component", "fitness", "diversity"]
            )
    except Exception as e:
        logger.error(f"Failed to create results CSV: {e}")

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
):
    """Run the genetic algorithm for n generations."""
    results_dir = os.path.dirname(results_path)
    time_elapse_path = os.path.join(results_dir, f"time_elapse_{exp_name}.csv") if results_dir else f"time_elapse_{exp_name}.csv"
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    population_size = len(population.units)

    with open(time_elapse_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["generation", "time_seconds"])

    for generation in range(num_generations):
        start_time = time.time()
        logger.info(f"================== Generation {generation} ==================")

        _evaluate_fitness(
            exp_name, generation, population, client, terrain, model_name,
            results_path=results_path, seed=seed, timesteps=timesteps, top_k=top_k,
        )

        # Selection step: keep top designs
        population.units = select_top_units(population.units, population_size)

        elapsed = time.time() - start_time
        logger.info(f"Generation {generation} completed in {elapsed:.1f}s")

        with open(time_elapse_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([generation, f"{elapsed:.2f}"])

        try:
            with open(results_path, "a", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                for i, unit in enumerate(population.units):
                    writer.writerow([
                        generation,
                        i,
                        f"evo{generation}_unit{i}.xml",
                        unit.structure,
                        unit.components,
                        unit.fitness,
                        unit.diversity,
                    ])
        except Exception as e:
            logger.error(f"Failed to write results CSV: {e}")

    return population


def generate_grammar(client: OpenAI, system_prompt: str, user_prompt: str, model_name: str):
    """Prompt the LLM with the robot grammar and return the raw response."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


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
) -> Population:
    """Evaluate fitness: LLM design -> compile -> train -> update population."""
    logger.info("Starting fitness evaluation...")
    start_time = time.time()

    results_dir = os.path.dirname(results_path) or "."

    elite_fitness = -1

    population_file = os.path.join(results_dir, f"population_{exp_name}.txt")
    curr_population = read_txt(population_file) if os.path.exists(population_file) else ""

    grammar_system = read_txt(prompts_path("grammar.txt"))

    current_elite = None
    xml_paths = []
    unit_records = []

    xml_exp_dir = os.path.join(results_dir, f"examples_{exp_name}")
    os.makedirs(xml_exp_dir, exist_ok=True)

    for i, unit in enumerate(population.units):
        # Reset unit state for this generation's new design
        unit.fitness = -1
        unit.xml_path = None

        grammar = generate_grammar(
            client, grammar_system,
            unit.prompt + " " + curr_population, model_name,
        )
        structural_graph = extract_last_step(grammar)
        component_graph = extract_bracket_line(grammar)
        diversity = None

        if component_graph is not None:
            diversity = design_score(structural_graph, component_graph)
            logger.info(f"unit{i}: {component_graph}")
            try:
                robot_xml = make_robot(compile_robot(component_graph))
                xml_path = os.path.join(xml_exp_dir, f"evo{generation}_unit{i}.xml")
                with open(xml_path, "w") as f:
                    f.write(robot_xml)
                if xml_has_joints(xml_path):
                    xml_paths.append(xml_path)
                    unit.xml_path = xml_path
            except Exception as e:
                logger.warning(f"Failed to compile unit {i}: {e}")

        unit.structure = structural_graph
        unit.components = component_graph
        unit.diversity = diversity

    # Train multiple models in parallel
    results = train_multiple_models(
        xml_paths,
        terrain=terrain,
        timesteps=timesteps,
        seed=seed,
    )
    if results is None:
        logger.warning("Training timed out, no results returned")
        results = []
    results_dict = dict(results)

    for unit in population.units:
        if unit.xml_path in results_dict:
            unit.fitness = results_dict[unit.xml_path]

        if unit.fitness > elite_fitness:
            current_elite = unit.model_copy()
            elite_fitness = unit.fitness

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
