import os
from pathlib import Path
from typing import List, Optional, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
from pandas import DataFrame
from pandas.core.groupby import GroupBy

import colors
from text.consts.columns import RANK_COL, DATASET_COL
from text.outlier_detection.odm import METHOD_COL


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

        self.create_box_plot(data, method_col, metric_col, [group_by])

    def create_box_plot(self, data: pd.DataFrame, method_col: str, metric_col: str, group_by: str or list, name: Optional[str]= None, method_color: Callable[[str], str] = None):
        # Use a modern style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Handle both single column and list of columns for group_by
        if isinstance(group_by, list):
            # Get unique combinations of values in the groupby columns
            groups = list(data.groupby(group_by).groups.keys())
            # Sort groups for consistent display
            groups = sorted(groups, key=str)
        else:
            groups = sorted(data[group_by].unique())

        n_groups = len(groups)
        unique_methods = data[method_col].unique()

        def default_color_mapping(method: str) -> str:
            if "VMMD" in method:
                return colors.VGAN_GREEN
            elif "FeatureBagging" in method:
                return colors.TRIADIC[0]
            else:
                return colors.TRIADIC[1]

        # Create custom color mapping based on method names
        color_map = {}
        color_func = default_color_mapping if method_color is None else method_color
        for method in unique_methods:
            color_map[method] = color_func(method)



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
            # Filter data based on whether group_by is a list or single column
            if isinstance(group_by, list):
                # For tuple of values from multiple columns
                mask = pd.Series(True, index=data.index)
                for col, val in zip(group_by, group):
                    mask &= (data[col] == val)
                group_data = data[mask]
            else:
                # Original behavior for single column
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

            # Create title based on group_by type
            if isinstance(group_by, list):
                title = ", ".join([f"{col}={val}" for col, val in zip(group_by, group)])
            else:
                title = f"{group}"

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel('')
            ax.set_xlabel('Rank', fontsize=12)

            # Improve appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', linestyle='--', alpha=0.6)

            # Ensure method names are fully visible
            ax.tick_params(axis='y', labelsize=11)
            ax.tick_params(axis='x', labelsize=11)

            # Set y-tick labels to be fully visible
            plt.setp(ax.get_yticklabels(), wrap=True)

            for i, artist in enumerate(ax.artists):
                artist.set_edgecolor('black')
                artist.set_linewidth(0.8)

        # Create a descriptive filename
        if isinstance(group_by, list):
            group_str = "_".join(group_by)
        else:
            group_str = group_by

        name = f"rank_{method_col}_{metric_col}_{group_str}" if name is None else name
        plt.savefig(self.output_dir / (name + ".png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_box_plot_vertical(self, data: pd.DataFrame, method_col: str, metric_col: str, group_by: str or list,
                                 name: Optional[str] = None, method_color: Callable[[str], str] = None,
                                 n_rows: Optional[int] = None, n_cols: Optional[int] = None):
        """
        Creates vertical box plots comparing method ranks within specified groups in a grid layout.
        Only includes methods present in all groups.

        :param data: DataFrame containing the data
        :param method_col: Column specifying the method name
        :param metric_col: Column specifying the metric
        :param group_by: Column(s) to group by
        :param name: Optional custom name for the output file
        :param method_color: Optional function to determine colors for methods
        :param n_rows: Number of rows in the plot grid (default: auto-calculated)
        :param n_cols: Number of columns in the plot grid (default: auto-calculated)
        """
        plt.style.use('seaborn-v0_8-whitegrid')

        # Determine groups
        if isinstance(group_by, list):
            groups = sorted(list(data.groupby(group_by).groups.keys()), key=str)
        else:
            groups = sorted(data[group_by].unique())

        common_methods = set(data[method_col].unique())
        filtered_data = data[data[method_col].isin(common_methods)].copy()
        unique_methods = sorted(common_methods)
        n_groups = len(groups)

        # Determine grid layout
        if n_rows is None and n_cols is None:
            n_rows, n_cols = n_groups, 1
        elif n_rows is None:
            n_rows = (n_groups + n_cols - 1) // n_cols
        elif n_cols is None:
            n_cols = (n_groups + n_rows - 1) // n_rows

        if n_rows * n_cols < n_groups:
            print(f"Warning: Grid size ({n_rows}x{n_cols}) too small for {n_groups} groups.")
            groups = groups[:n_rows * n_cols]

        # Color mapping
        def default_color_mapping(method: str) -> str:
            if "VMMD" in method:
                return colors.VGAN_GREEN
            elif "FeatureBagging" in method:
                triadic_0 = colors.TRIADIC[0]
                return triadic_0 if triadic_0.startswith('#') else f"#{triadic_0}"
            else:
                triadic_1 = colors.TRIADIC[1]
                return triadic_1 if triadic_1.startswith('#') else f"#{triadic_1}"

        color_func = default_color_mapping if method_color is None else method_color
        color_map = {m: color_func(m) for m in unique_methods}

        fig_w = max(5, n_cols * len(unique_methods) * 0.4)
        fig_h = max(4, n_rows * 2)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                                 sharey=True, constrained_layout=True)

        # Normalize axes array
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Ensure CSV output dir
        csv_path = self.output_dir / "rank_csv"
        csv_path.mkdir(parents=True, exist_ok=True)

        axes_flat = axes.flatten()
        for ax, group in zip(axes_flat, groups):
            # Filter per‐group
            if isinstance(group_by, list):
                mask = pd.Series(True, index=filtered_data.index)
                for col, val in zip(group_by, group):
                    mask &= filtered_data[col] == val
                group_df = filtered_data[mask]
                space_val = group[1]
            else:
                group_df = filtered_data[filtered_data[group_by] == group]
                space_val = None

            # Compute ranks
            reranked = group_df.copy()
            reranked[RANK_COL] = group_df.groupby(DATASET_COL)[metric_col] \
                .rank(ascending=False, method='min')

            # Select methods for this subplot
            methods_plot = unique_methods.copy()
            if space_val == "Avg":
                methods_plot = [m for m in methods_plot if "Text" not in m]
            elif space_val == "NPTE":
                methods_plot = [m for m in methods_plot if "Token" not in m]

            plot_df = reranked[reranked[method_col].isin(methods_plot)]
            plot_palette = {m: color_map[m] for m in methods_plot}

            # Draw boxplot
            seaborn.boxplot(
                x=method_col, y=RANK_COL, data=plot_df, ax=ax,
                hue=method_col, palette=plot_palette,
                order=methods_plot, width=0.7, legend=False, orient='v'
            )

            # Title & labels
            title = ", ".join(map(str, group)) if isinstance(group_by, list) else str(group)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Rank', fontsize=12)

            # Styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', labelsize=10, rotation=45)
            ax.tick_params(axis='y', labelsize=10)
            for artist in ax.artists:
                artist.set_edgecolor('black')
                artist.set_linewidth(0.8)

        # Hide extra subplots
        for ax in axes_flat[len(groups):]:
            ax.set_visible(False)

        # Save
        group_str = "_".join(group_by) if isinstance(group_by, list) else group_by
        base = name or f"rank_{method_col}_{metric_col}_{group_str}"
        fname = f"{base}_grid_{n_rows}x{n_cols}.pdf"
        plt.savefig(self.output_dir / fname, dpi=300, bbox_inches='tight')
        plt.close()


def manual_ranking():
    from src.text.outlier_detection.odm import METHOD_COL, AUC_COL, BASE_COL, SPACE_COL
    from src.text.visualizer.result_visualizer.result_aggregator import coloring

    version_path = Path(__file__).parent.parent.parent.parent.parent / "experiments" / "0.45" / "rank_csv"
    ranked_df = pd.read_csv(version_path / "ranked_results_filterd_renamed.csv")
    ranked_df[METHOD_COL] = ranked_df[METHOD_COL].replace("Feature-Bagging", "FB").replace("V-GAN-Token", "V-GAN-\nToken").replace("V-GAN-Text", "V-GAN-\nText").replace("Full-Space", "Full")
    ranked_df = ranked_df[ranked_df[BASE_COL] != "ECOD"]
    ranked_df = ranked_df[ranked_df[METHOD_COL] != "Trivial"]

    rank = RankVisualizer([], version_path)

    rank.create_box_plot_vertical(data=ranked_df,
                                  method_col=METHOD_COL,
                                  metric_col=AUC_COL,
                                  group_by=[BASE_COL, SPACE_COL],
                                  method_color=coloring, name="ranked_renamed_vertical", n_cols=2)

if __name__ == "__main__":
    manual_ranking()