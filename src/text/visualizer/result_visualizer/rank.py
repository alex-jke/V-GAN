import os
from pathlib import Path
from typing import List, Optional, Callable

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
                                 name: Optional[str] = None, method_color: Callable[[str], str] = None):
        """
        Creates vertical box plots comparing method ranks within specified groups.
        Only includes methods present in all groups.
        """
        # Use a modern style
        plt.style.use('seaborn-v0_8-whitegrid')

        # --- Start: Find common methods across groups ---
        if isinstance(group_by, list):
            groups = list(data.groupby(group_by).groups.keys())
            groups = sorted(groups, key=str)
        else:
            groups = sorted(data[group_by].unique())

        common_methods = None
        for group in groups:
            if isinstance(group_by, list):
                mask = pd.Series(True, index=data.index)
                for col, val in zip(group_by, group):
                    mask &= (data[col] == val)
                group_data = data[mask]
            else:
                group_data = data[data[group_by] == group]

            methods_in_group = set(group_data[method_col].unique())
            if common_methods is None:
                common_methods = methods_in_group
            else:
                common_methods &= methods_in_group

        if not common_methods:
            print(f"Warning: No methods found present in all groups for {group_by}. Skipping plot.")
            return

        filtered_data = data[data[method_col].isin(common_methods)].copy()
        unique_methods = sorted(list(common_methods))
        # --- End: Find common methods across groups ---

        n_groups = len(groups)

        def default_color_mapping(method: str) -> str:
            if "VMMD" in method:
                return colors.VGAN_GREEN
            elif "FeatureBagging" in method:
                # Ensure hex color has '#' prefix
                triadic_0 = colors.TRIADIC[0]
                return triadic_0 if triadic_0.startswith('#') else f"#{triadic_0}"
            else:
                # Ensure hex color has '#' prefix
                triadic_1 = colors.TRIADIC[1]
                return triadic_1 if triadic_1.startswith('#') else f"#{triadic_1}"

        # Create custom color mapping based on method names
        color_map = {}
        color_func = default_color_mapping if method_color is None else method_color
        for method in unique_methods:
            color_map[method] = color_func(method)

        # Dynamically size figure based on content
        # Adjust width based on number of methods, height based on groups
        fig_width = max(12, len(unique_methods) * 1.5)  # Wider for vertical boxes
        fig_height = max(8, n_groups * 4)  # Taller per subplot

        # Create figure with subplots stacked vertically
        fig, axes = plt.subplots(n_groups, 1, figsize=(fig_width, fig_height),
                                 sharey=True, constrained_layout=True)  # Share Y (Rank) axis

        # Handle single group case
        axes = [axes] if n_groups == 1 else axes

        # Create CSV directory if needed
        csv_path = self.output_dir / "rank_csv"
        if not csv_path.exists():
            csv_path.mkdir(parents=True)

        for ax, group in zip(axes, groups):
            # Filter data based on whether group_by is a list or single column
            # Use filtered_data here
            if isinstance(group_by, list):
                mask = pd.Series(True, index=filtered_data.index)
                for col, val in zip(group_by, group):
                    mask &= (filtered_data[col] == val)
                group_data = filtered_data[mask]
            else:
                group_data = filtered_data[filtered_data[group_by] == group]

            # Create vertical boxplot with methods on x-axis
            seaborn.boxplot(
                x=method_col,  # Methods on X-axis
                y=RANK_COL,  # Rank on Y-axis
                data=group_data,
                ax=ax,
                hue=method_col,  # Color by method
                palette=color_map,
                order=unique_methods,  # Ensure consistent method order
                width=0.7,
                legend=False,
                # orient='v' is default, no need to specify
            )

            # Create title based on group_by type
            if isinstance(group_by, list):
                title = ", ".join([f"{col}={val}" for col, val in zip(group_by, group)])
            else:
                title = f"{group}"

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('')  # Remove x-axis label (methods shown as ticks)
            ax.set_ylabel('Rank', fontsize=12)

            # Improve appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', linestyle='--', alpha=0.6)  # Grid lines for Y (Rank) axis

            # Rotate x-axis labels for better visibility
            ax.tick_params(axis='x', labelsize=11, rotation=45, ha='right')
            ax.tick_params(axis='y', labelsize=11)

            # Optional: Add edge color to boxes (already present in your original code)
            for i, artist in enumerate(ax.artists):
                artist.set_edgecolor('black')
                artist.set_linewidth(0.8)

        # Create a descriptive filename
        if isinstance(group_by, list):
            group_str = "_".join(group_by)
        else:
            group_str = group_by

        # Add '_vertical' to distinguish from the horizontal plot version
        base_name = f"rank_{method_col}_{metric_col}_{group_str}" if name is None else name
        plot_name = f"{base_name}_vertical"
        plt.savefig(self.output_dir / (plot_name + ".png"),
                    dpi=300, bbox_inches='tight')
        plt.close()