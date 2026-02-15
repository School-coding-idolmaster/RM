# RoboMorph: Evolving Robot Morphology using Large Language Models

[![Paper](https://img.shields.io/badge/arXiv-2407.08626-b31b1b.svg)](https://arxiv.org/abs/2407.08626)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![ICRA 2026](https://img.shields.io/badge/ICRA-2026-orange.svg)](https://arxiv.org/abs/2407.08626)

**RoboMorph** automatically discovers diverse, high-performing robot morphologies by combining large language models with evolutionary search. Given a task description, it evolves robot body plans across generations without any manual design.

<p align="center">
  <img src="assets/morphologies.png" alt="Diverse robot morphologies discovered by RoboMorph" width="100%">
</p>
<p align="center"><i>Diverse robot morphologies automatically discovered by RoboMorph across different terrains.</i></p>

## How It Works

RoboMorph uses an LLM-guided evolutionary loop to design robots:

1. **Design**: An LLM generates robot designs expressed in a structured grammar.
2. **Compile**: Each design is compiled into a [MuJoCo](https://mujoco.org/) XML model.
3. **Train**: The robot is trained with reinforcement learning (SAC or PPO) using [Brax](https://github.com/google/brax) on GPU-accelerated parallel environments.
4. **Evaluate**: Each robot is scored on its locomotion fitness (forward velocity).
5. **Evolve**: The top-performing designs are fed back to the LLM as few-shot examples, guiding the next generation toward better morphologies.

<p align="center">
  <img src="assets/pipeline.png" alt="RoboMorph pipeline" width="100%">
</p>

This loop repeats for many generations, producing increasingly capable robots across four terrain types: ridged, flat, frozen lake, and beams.

<p align="center">
  <img src="assets/top_designs.png" alt="Top-performing robot designs across terrains" width="100%">
</p>
<p align="center"><i>Top-performing robot designs discovered by RoboMorph on each terrain.</i></p>

## Quick Start

### 1. Install

We recommend [uv](https://docs.astral.sh/uv/) for fast, reproducible setup:

```bash
git clone https://github.com/kevinxqiu/robomorph.git
cd robomorph

# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install all dependencies (locked versions)
uv sync
```

<details>
<summary>Alternative: pip</summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

</details>

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

- **`OPENAI_API_KEY`** (required): Used for LLM-based design generation.
- **`WANDB_API_KEY`** (optional): Enables experiment logging via [Weights & Biases](https://wandb.ai).

### 3. Run

```bash
robomorph --exp_name my_experiment --terrain flat
```

This evolves a population of 32 robots over 100 generations on flat terrain using GPT-4o. Results are saved to `results/results_my_experiment.csv`.

## Usage

```bash
robomorph \
    --exp_name <experiment_name> \
    --terrain <flat|ridged|beams|frozen_lake> \
    --population_size 32 \
    --num_generations 50 \
    --model gpt-4o
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp_name` | (required) | Experiment name for logging and output files |
| `--terrain` | `flat` | Terrain type: `flat`, `ridged`, `beams`, `frozen_lake` |
| `--population_size` | `32` | Number of robot designs per generation |
| `-n`, `--num_generations` | `100` | Number of evolutionary generations |
| `--model` | `gpt-4o` | OpenAI model used for design generation |
| `--timesteps` | `1000000` | RL training timesteps per robot |
| `--top_k` | `3` | Number of top designs kept as few-shot examples |
| `--seed` | `42` | Random seed for reproducibility |
| `--results_path` | `results/results_<exp>.csv` | Output path for the results CSV |
| `-p`, `--problem` | from `prompts/user.txt` | Task description prompt sent to the LLM |

### Reproducing Paper Results

To reproduce the main experiments from the paper, run each terrain with the default settings:

```bash
robomorph --exp_name flat_run      --terrain flat
robomorph --exp_name ridged_run    --terrain ridged
robomorph --exp_name beams_run     --terrain beams
robomorph --exp_name frozen_run    --terrain frozen_lake
```

Each run uses a population of 32 robots, 100 generations, 1M training timesteps per robot, and the top 3 designs as few-shot context. These match the default argument values. All results are written to the `results/` directory.

### Training Individual Robots

You can train a single robot XML directly using SAC or PPO:

```bash
python -m train.train_brax --xml train/control_xmls/ant.xml --terrain flat --algorithm sac
```

Run `python -m train.train_brax --help` for the full list of training arguments.

## Project Structure

```
robomorph/                     # Main package
    main.py                    # Entry point and CLI
    evolution/                 # Evolutionary algorithm
        __init__.py            # Public API re-exports
        loop.py                # Core evolution loop
        population.py          # EvolutionUnit and Population (Pydantic models)
        utils.py               # Dataset I/O, merging, diversity scoring
        plots.py               # Visualization utilities
    prompts/
        grammar.txt            # Robot grammar specification
        user.txt               # Default task description prompt
compiler/                      # Grammar-to-XML compiler
    compile.py                 # Tokenizer, compiler, and MuJoCo XML generator
    robot_generator.py         # Random robot generation from grammar
train/                         # RL training infrastructure
    train_brax.py              # SAC/PPO trainer (Brax + JAX)
    parallel_train.py          # Multi-GPU parallel training
    render_policy.py           # Policy visualization (HTML)
    envs/                      # Terrain environments (flat, ridged, beams, frozen_lake)
    control_xmls/              # Baseline robot XML (ant)
```

## Citation

If you use RoboMorph in your research, please cite:

```bibtex
@article{qiu2024robomorph,
  title={Robomorph: Evolving robot morphology using large language models},
  author={Qiu, Kevin and Pa{\l}ucki, W{\l}adys{\l}aw and Ciebiera, Krzysztof and Fija{\l}kowski, Pawe{\l} and Cygan, Marek and Kuci{\'n}ski, {\L}ukasz},
  journal={arXiv preprint arXiv:2407.08626},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
