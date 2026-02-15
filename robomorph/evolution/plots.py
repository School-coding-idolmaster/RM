import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Z_TABLE = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}


def confidence_interval(mean, std, n, confidence=0.95):
    """Compute confidence interval bounds for the given confidence level."""
    z = Z_TABLE.get(confidence, 1.96)
    margin_of_error = z * (std / np.sqrt(n))
    return mean - margin_of_error, mean + margin_of_error


def compute_evolution_stats(file_path):
    """Compute per-generation statistics from a results CSV file."""
    df = pd.read_csv(file_path, delimiter=";", on_bad_lines='warn', engine='python')

    # Support both old format (robot column) and new format (generation column)
    if 'generation' in df.columns:
        df['evolution'] = df['generation']
    else:
        df['evolution'] = df['robot'].apply(lambda x: int(x.split('_')[0][3:]) if 'evo' in x else None)

    grouped = df.groupby('evolution')

    stats_df = pd.DataFrame({
        'evolution': list(grouped.groups.keys()),
        'max_fitness': grouped['fitness'].max().values,
        'avg_fitness': grouped['fitness'].mean().values,
        'std_fitness': grouped['fitness'].std().values,
        'max_diversity': grouped['diversity'].max().values,
        'avg_diversity': grouped['diversity'].mean().values,
        'std_diversity': grouped['diversity'].std().values,
        'n_fitness': grouped['fitness'].count().values,
        'n_diversity': grouped['diversity'].count().values,
    })
    stats_df['fitness_ci_lower'], stats_df['fitness_ci_upper'] = confidence_interval(
        stats_df['avg_fitness'], stats_df['std_fitness'], stats_df['n_fitness'])
    stats_df['diversity_ci_lower'], stats_df['diversity_ci_upper'] = confidence_interval(
        stats_df['avg_diversity'], stats_df['std_diversity'], stats_df['n_diversity'])

    return stats_df


def plot_max_fitness(evolution, max_fitness, save_path=None):
    plt.figure()
    plt.plot(evolution, max_fitness, marker='o', linestyle='-', color='b')
    plt.xlabel('Evolution')
    plt.ylabel('Max Fitness')
    plt.title('Max Fitness per Evolution')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format='png')
    plt.close()


def plot_avg_fitness_ci(evolution, avg_fitness, ci_lower, ci_upper, save_path=None):
    plt.figure()
    plt.plot(evolution, avg_fitness, marker='o', linestyle='-', color='g', label='Avg Fitness')
    plt.fill_between(evolution, ci_lower, ci_upper, color='g', alpha=0.2, label='95% CI')
    plt.xlabel('Evolution')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness per Evolution (with 95% CI)')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format='png')
    plt.close()


def plot_max_diversity(evolution, max_diversity, save_path=None):
    plt.figure()
    plt.plot(evolution, max_diversity, marker='o', linestyle='-', color='r')
    plt.xlabel('Evolution')
    plt.ylabel('Max Diversity')
    plt.title('Max Diversity per Evolution')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format='png')
    plt.close()


def plot_avg_diversity_ci(evolution, avg_diversity, ci_lower, ci_upper, save_path=None):
    plt.figure()
    plt.plot(evolution, avg_diversity, marker='o', linestyle='-', color='m', label='Avg Diversity')
    plt.fill_between(evolution, ci_lower, ci_upper, color='m', alpha=0.2, label='95% CI')
    plt.xlabel('Evolution')
    plt.ylabel('Average Diversity')
    plt.title('Average Diversity per Evolution (with 95% CI)')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format='png')
    plt.close()


def plot_multiple_max_fitness(evolution_list, max_fitness_list, labels, save_path=None):
    plt.figure()
    for i, (evolution, max_fitness) in enumerate(zip(evolution_list, max_fitness_list)):
        plt.plot(evolution, max_fitness, marker='o', linestyle='-', label=labels[i])
    plt.xlabel('Evolution')
    plt.ylabel('Max Fitness')
    plt.title('Max Fitness per Evolution')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format='png')
    plt.close()


def plot_multiple_avg_fitness_ci(evolution_list, avg_fitness_list, ci_lower_list, ci_upper_list, labels, save_path=None):
    plt.figure()
    for i, (evolution, avg_fitness, ci_lower, ci_upper) in enumerate(zip(evolution_list, avg_fitness_list, ci_lower_list, ci_upper_list)):
        plt.plot(evolution, avg_fitness, marker='o', linestyle='-', label=labels[i])
        plt.fill_between(evolution, ci_lower, ci_upper, alpha=0.2)
    plt.xlabel('Evolution')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness per Evolution (with 95% CI)')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format='png')
    plt.close()


def plot(csv_path, results_dir='../../results/'):
    """Plot single-experiment statistics from a results CSV file."""
    stats_df = compute_evolution_stats(csv_path)

    evolution = stats_df['evolution']
    fitness_ci_lower = stats_df['fitness_ci_lower']
    fitness_ci_upper = stats_df['fitness_ci_upper']
    diversity_ci_lower = stats_df['diversity_ci_lower']
    diversity_ci_upper = stats_df['diversity_ci_upper']

    os.makedirs(results_dir, exist_ok=True)

    plot_max_fitness(evolution, stats_df['max_fitness'], save_path=os.path.join(results_dir, 'max_fitness.png'))
    plot_avg_fitness_ci(evolution, stats_df['avg_fitness'], fitness_ci_lower, fitness_ci_upper, save_path=os.path.join(results_dir, 'avg_fitness_ci.png'))
    plot_max_diversity(evolution, stats_df['max_diversity'], save_path=os.path.join(results_dir, 'max_diversity.png'))
    plot_avg_diversity_ci(evolution, stats_df['avg_diversity'], diversity_ci_lower, diversity_ci_upper, save_path=os.path.join(results_dir, 'avg_diversity_ci.png'))


def plot_comparison(csv_files, labels=None, results_dir='../../results/'):
    """Plot comparison statistics across multiple experiment CSV files.

    Args:
        csv_files: List of paths to results CSV files.
        labels: Display labels for each file. Derived from filenames if not provided.
        results_dir: Directory to save output plots.
    """
    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

    evolution_list = []
    max_fitness_list = []
    avg_fitness_list = []
    ci_lower_list = []
    ci_upper_list = []

    for file_path in csv_files:
        stats_df = compute_evolution_stats(file_path)
        evolution_list.append(stats_df['evolution'])
        max_fitness_list.append(stats_df['max_fitness'])
        avg_fitness_list.append(stats_df['avg_fitness'])
        ci_lower_list.append(stats_df['fitness_ci_lower'])
        ci_upper_list.append(stats_df['fitness_ci_upper'])

    os.makedirs(results_dir, exist_ok=True)

    plot_multiple_max_fitness(evolution_list, max_fitness_list, labels, save_path=os.path.join(results_dir, 'max_fitness_comparison.png'))
    plot_multiple_avg_fitness_ci(evolution_list, avg_fitness_list, ci_lower_list, ci_upper_list, labels, save_path=os.path.join(results_dir, 'avg_fitness_ci_comparison.png'))
