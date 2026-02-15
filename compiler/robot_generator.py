import random


def apply_r0():
    return "S"

def apply_r1(structure):
    return structure.replace("S", "H-B-T", 1)

def apply_r2(structure):
    return structure.replace("T", "Y-B-T", 1)

def apply_r3(structure):
    """Randomly replace one 'B' node with a leg sub-tree."""
    b_positions = [i for i, char in enumerate(structure) if char == 'B']
    if b_positions:
        selected_position = random.choice(b_positions)
        structure = structure[:selected_position] + "U-(C-M-E)" + structure[selected_position + 1:]
    return structure

def apply_r4(structure):
    return structure.replace("B", "U", 1)

def apply_r5(structure):
    """Randomly replace one 'E' node with an extended limb."""
    e_positions = [i for i, char in enumerate(structure) if char == 'E']
    if e_positions:
        selected_position = random.choice(e_positions)
        structure = structure[:selected_position] + "J-L-E" + structure[selected_position + 1:]
    return structure

def apply_r6(structure):
    return structure.replace("T", "C-M-E", 1)

def apply_r7(structure):
    return structure.replace("H", "E-M-C", 1)


def random_design_robot():
    """Generate a random robot structural graph using grammar rules."""
    rules = [apply_r1, apply_r2, apply_r3, apply_r4, apply_r5, apply_r6, apply_r7]
    rule_names = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7']
    weights = [1, 1, 1, 1, 1, 0.01, 0.01]
    structure_steps = []

    structure = apply_r0()
    structure_steps.append(f"Step 1: r0) {structure}")
    step = 2
    max_steps = 15

    while step <= max_steps:
        applicable_rules = [
            (rule, rule_name, weight)
            for rule, rule_name, weight in zip(rules, rule_names, weights)
            if rule(structure) != structure
        ]

        if not applicable_rules:
            break

        filtered_rules, filtered_rule_names, filtered_weights = zip(*applicable_rules)

        chosen_index = random.choices(range(len(filtered_rules)), weights=filtered_weights, k=1)[0]
        chosen_rule = filtered_rules[chosen_index]
        chosen_rule_name = filtered_rule_names[chosen_index]

        structure = chosen_rule(structure)
        structure_steps.append(f"Step {step}: {chosen_rule_name}) {structure}")
        step += 1

    # Apply r4 to remove all remaining 'B' nodes
    while 'B' in structure:
        structure = apply_r4(structure)
        structure_steps.append(f"Step {step}: r4) {structure}")
        step += 1

    return structure_steps, structure


def convert_to_components(structure):
    """Convert a structural graph into a component graph with random parameters."""
    component_rules = {
        'U': lambda: f"body=15cm",
        'L': lambda: f"limb={random.choice([10, 15])}cm",
        'Y': lambda: random.choice(["rigid", "roll", "twist"]),
        'J': lambda: random.choice([
            "rigid",
            "roll",
            f"knee={random.randint(0, 60)}deg",
            f"elbow={random.randint(0, 180)}deg"
        ]),
        'C': lambda: "connector",
        'M': lambda: "mount",
        'E': lambda: random.choice(["wheel", "null"]),
        'H': lambda: "null",
        'T': lambda: "null"
    }

    result = []
    i = 0
    while i < len(structure):
        char = structure[i]
        if char == '(':
            substructure = []
            depth = 1
            i += 1
            while depth > 0:
                if structure[i] == '(':
                    depth += 1
                elif structure[i] == ')':
                    depth -= 1
                if depth > 0:
                    substructure.append(structure[i])
                i += 1
            subcomponents = convert_to_components(''.join(substructure))
            result.append(f"({subcomponents})")
        elif char == '-':
            if result and result[-1] != '-':
                result.append('-')
            i += 1
        else:
            if char in component_rules:
                result.append(f"[{component_rules[char]()}]")
            i += 1

    if result and result[-1] == '-':
        result.pop()

    return ''.join(result)


def create_init_few_shots(population_size):
    """Create initial few-shot examples for the LLM grammar prompt."""
    structural_rules = []
    component_rules = []
    for _ in range(population_size):
        structure_steps, structure = random_design_robot()
        converted_components = convert_to_components(structure)
        structural_rules.append(structure_steps)
        component_rules.append(converted_components)
    return structural_rules, component_rules


def generate_init_few_shots(population_size, output_file):
    """Generate few-shot examples and write them to a file."""
    structural_rules, component_rules = create_init_few_shots(population_size)

    with open(output_file, 'w') as f:
        for i in range(population_size):
            f.write(f"### EXAMPLE {i + 1} ###\n")
            f.write("STRUCTURAL RULES:\n")
            for step in structural_rules[i]:
                f.write(f"{step}\n")
            f.write("\nCOMPONENT RULES:\n")
            f.write(f"{component_rules[i]}\n")
            f.write("\nREASONING: n/a\n\n")
            f.write("FITNESS: -1\n\n")

    return component_rules


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate initial robot few-shot examples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--population_size", type=int, default=5, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="test.txt", help="Output file path")
    args = parser.parse_args()

    random.seed(args.seed)
    generate_init_few_shots(args.population_size, args.output)
