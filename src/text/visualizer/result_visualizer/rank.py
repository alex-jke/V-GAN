import os
from pathlib import Path
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
import seaborn

import colors
from text.consts.columns import RANK_COL

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
            df_run[RANK_COL] = df_run.groupby(group_by)[metric_col].rank(ascending=False, method='min')
            dfs.append(df_run)
        data = pd.concat(dfs, ignore_index=True)

        self.create_box_plot(data, method_col, metric_col, group_by)

    def create_box_plot(self, data: pd.DataFrame, method_col: str, metric_col: str, group_by: str):
        # Use a modern style
        plt.style.use('seaborn-v0_8-whitegrid')


        # Identify unique groups and methods
        groups = sorted(data[group_by].unique())
        n_groups = len(groups)
        unique_methods = data[method_col].unique()

        # Create custom color mapping based on method names
        color_map = {}
        for method in unique_methods:
            if "VMMD" in method:
                color_map[method] = colors.VGAN_GREEN
            elif "FeatureBagging" in method:
                color_map[method] = colors.TRIADIC[0]
            else:
                color_map[method] = colors.TRIADIC[1]

        # Dynamically size figure based on content
        fig_width = max(12, len(max(unique_methods, key=len)) * 0.3)
        fig_height = max(8, n_groups * 2.5 + len(unique_methods) * 0.5)

        # Create figure with subplots stacked vertically
        fig, axes = plt.subplots(n_groups, 1, figsize=(fig_width, fig_height),
                                sharex=True, constrained_layout=True)

        # Handle single group case
        axes = [axes] if n_groups == 1 else axes

        # Create CSV directory if needed
        csv_path = self.output_dir / "rank_csv"
        if not csv_path.exists():
            csv_path.mkdir(parents=True)

        for ax, group in zip(axes, groups):
            group_data = data[data[group_by] == group]

            # Create horizontal boxplot with methods on y-axis
            seaborn.boxplot(
                x=RANK_COL,
                y=method_col,
                data=group_data,
                ax=ax,
                hue=method_col,
                palette=color_map,
                width=0.7,
                legend=False,
                orient='h'
            )

            # Add group title and style axis labels
            ax.set_title(f'{group}', fontsize=14, fontweight='bold')
            ax.set_ylabel('')
            ax.set_xlabel('Rank', fontsize=12)

            # Improve appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', linestyle='--', alpha=0.6)

            # Ensure method names are fully visible
            ax.tick_params(axis='y', labelsize=11)
            ax.tick_params(axis='x', labelsize=11)

            # Set y-tick labels to be fully visible with increased margin
            plt.setp(ax.get_yticklabels(), wrap=True)

            for i, artist in enumerate(ax.artists):
                artist.set_edgecolor('black')
                artist.set_linewidth(0.8)

        plt.savefig(self.output_dir / f"rank_{method_col}_{metric_col}_{group_by}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()