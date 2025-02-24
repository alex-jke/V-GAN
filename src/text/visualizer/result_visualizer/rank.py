import os
from pathlib import Path
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
import seaborn

class RankVisualizer():
    """
    A class to visualize the rank of methods in comparison to each other with regard to a metric.
    A box plot is created to show the distribution of ranks per method.
    """

    def __init__(self, results: List[pd.DataFrame], output_dir: Path):
        self.results: List[pd.DataFrame] = results
        self.output_dir: Path = output_dir

    def visualize(self, method_col: str, metric_col: str, group_by: str):
        """
        Visualizes the rank the methods attain in comparison to each other
        with regard to the metric specified, as a box plot.
        :param method_col: Column specifying the method.
        :param metric_col: Column specifying the metric.
        :param group_by: Column to group by. A separate subplot is created per group.
        """
        dfs = []
        for run in self.results:
            df_run = run.copy()
            df_run['rank'] = df_run.groupby(group_by)[metric_col].rank(ascending=False, method='min')
            dfs.append(df_run)
        data = pd.concat(dfs, ignore_index=True)

        # Identify unique groups and create a subplot for each
        groups = sorted(data[group_by].unique())
        n_groups = len(groups)
        fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5), sharey=True) if n_groups > 1 else (
            plt.subplots(1, 1, figsize=(5, 5)))
        if n_groups == 1:
            axes = [axes]

        # Create a boxplot for each group showing the distribution of ranks per method
        for ax, group in zip(axes, groups):
            group_data = data[data[group_by] == group]
            seaborn.boxplot(x=method_col, y='rank', data=group_data, ax=ax)
            ax.set_title(f'Group: {group}')
            ax.set_xlabel('Method')
            ax.set_ylabel('Rank')

        plt.tight_layout()

        # Save the resulting plot
        output_file = self.output_dir / f"rank_{method_col}_{metric_col}_{group_by}.png"
        plt.savefig(output_file)
        plt.close()