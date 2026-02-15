import os
import re

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts")


def prompts_path(filename: str) -> str:
    """Return the absolute path to a file in the robomorph/prompts/ directory."""
    return os.path.join(_PROMPTS_DIR, filename)


def read_txt(file_path):
    """Read and return the contents of a text file."""
    with open(file_path, 'r') as f:
        return f.read()


def convert_dataset_to_zip(text):
    """Parse population file text into (structural, component, fitness) tuples."""
    structural_pattern = re.compile(r'Structural: (.+)')
    components_pattern = re.compile(r'Components: (.+)')
    fitness_pattern = re.compile(r'Fitness: (-?[\d.]+)')

    structures = structural_pattern.findall(text)
    components = components_pattern.findall(text)
    fitness = [float(f) for f in fitness_pattern.findall(text)]

    return list(zip(structures, components, fitness))


def write_txt(file_path, dataset):
    """Write population dataset to a text file for LLM few-shot context."""
    with open(file_path, 'w') as f:
        f.write("Here are some examples of designs and their associated fitness values. "
                "How can you design a better robot with a higher fitness?\n")
        for i, entry in enumerate(dataset):
            f.write(f"Example {i}:\n")
            f.write(f"Structural: {entry[0]}\n")
            f.write(f"Components: {entry[1]}\n")
            f.write(f"Fitness: {entry[2]}\n\n")


def merge_datasets(dataset1, dataset2, k):
    """Combine old and new populations, keeping the top-k entries by fitness."""
    combined = list(dataset1) + list(dataset2)
    return sorted(combined, key=lambda x: x[2], reverse=True)[:k]


def diversity_score(counts):
    """Score diversity based on the number of unique non-zero component types.

    Uses an exponential bonus: score = n * 2^(n-1) where n = number of non-zero types.
    """
    non_zero = sum(1 for v in counts.values() if v > 0)
    if non_zero > 1:
        return non_zero * (2 ** (non_zero - 1))
    return non_zero


def design_score(structural_graph, component_graph):
    """Compute diversity score from structural and component graphs.

    Counts body segments, legs, and variety of joint types used.
    """
    if not isinstance(structural_graph, str):
        return 0

    num_body = structural_graph.count("U")
    num_legs = structural_graph.count("(")

    parts = re.findall(r'\[([^=\]]*).*?\]', component_graph)

    body_joints = {'rigid': 0, 'twist': 0, 'roll': 0}
    leg_joints = {'rigid': 0, 'roll': 0, 'knee': 0, 'elbow': 0}

    for i, part in enumerate(parts):
        try:
            if part == "body" and parts[i - 1] in body_joints:
                body_joints[parts[i - 1]] += 1
            if part == "mount" and parts[i + 1] in leg_joints:
                leg_joints[parts[i + 1]] += 1
        except (IndexError, KeyError):
            pass

    return num_body + num_legs + diversity_score(body_joints) + diversity_score(leg_joints)
