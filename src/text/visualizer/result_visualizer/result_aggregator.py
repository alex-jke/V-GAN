from pathlib import Path
import os
from typing import Optional, Callable

import pandas as pd

import colors
from text.consts.columns import EMB_MODEL_COL, DATASET_COL, RANK_COL
from text.outlier_detection.odm import METHOD_COL, SPACE_COL, AUC_COL, PRAUC_COL, F1_COL, TIME_TAKEN_COL, BASE_COL
from text.visualizer.result_visualizer.rank import RankVisualizer

FEATURE_BAGGING = "Feature-Bagging"
#V_GAN_AVG = "V-GAN-Avg"
#V_GAN_NPTE = "V-GAN-NPTE"
V_GAN = "V-GAN"
V_GAN_TOKEN = "V-GAN-Token"
V_GAN_TEXT = "V-GAN-Text"
FULL_SPACE = "Full-Space"
TRIVIAL = "Trivial"
AVG_SPACE = "Avg"
NTPE_SPACE = "NPTE"


def coloring(method_name: str) -> str:
    """
    Coloring function for the methods after renaming.
    """
    if "V-GAN" in method_name:
        return colors.VGAN_GREEN
    elif FEATURE_BAGGING in method_name or "FB" in method_name:
        return colors.TRIADIC[0]
    else:
        return colors.TRIADIC[1]

class ResultAggregator():

    def __init__(self, version_path: Path):
        """
        Initializes the ResultAggregator with the path to the version directory. This class will traverse the nested
        directory structure to find all the results stored in the subfolders. The results are expected to be in CSV format.
        Computes the average over the runs and stores the results in a new CSV file in the version directory.
        :param version_path: The path to the version directory. This should be the base directory where the results are
            stored in the subfolder structure.
        """
        self.version_path = version_path
        assert self.version_path.exists(), f"Version path {self.version_path} does not exist."

    def aggregate(self):
        """
        Aggregates the results from the subfolders in the version directory. This method will traverse the directory
        structure, find all the CSV files, and compute the average over the runs. The results are stored in a new CSV
        file in the version directory. The folder structure is expected to be as follows: version_path / [datasets] / [models] / run_[0..n] / result.csv
        """

        result_dfs = []

        # Traverse the directory structure to find all the CSV files
        for dataset_path in self.version_path.iterdir():
            if not dataset_path.is_dir():
                continue
            for model_path in dataset_path.iterdir():
                if not model_path.is_dir():
                    continue
                for run_path in model_path.iterdir():
                    if not run_path.is_dir() or not run_path.name.startswith("run_"):
                        continue

                    result_file = run_path / "results.csv"

                    model_name = model_path.name
                    dataset_name = dataset_path.name

                    if result_file.exists():
                        # Read the CSV file and compute the average over the runs
                        df = pd.read_csv(result_file)

                        df[EMB_MODEL_COL] = model_name
                        df[DATASET_COL] = dataset_name

                        result_dfs.append(df)

        group_by_columns = [METHOD_COL, SPACE_COL, BASE_COL, DATASET_COL, EMB_MODEL_COL]

        df = pd.concat(result_dfs, ignore_index=True)

        df = df[df[SPACE_COL] != "Word avg"]

        # Only include methods present in all runs
        amount_runs = len(result_dfs)
        included_methods = []
        for method in df[METHOD_COL].unique():
            present_in = len(df[df[METHOD_COL] == method].index)
            if present_in + 10 > amount_runs:
                included_methods.append(method)
            #else:
                #print(f"{method} not present in all runs ({present_in} / {amount_runs}).")

        # Include only the methods present in all runs or if the name contains "TextVMMD"
        df = df[df[METHOD_COL].isin(included_methods) | df[METHOD_COL].str.contains("TextVMMD")]

        # Average over the runs
        avg_df: pd.DataFrame = df.groupby(group_by_columns).mean().reset_index()

        # Group by dataset and create rankings from the average AUC, where the best AUC is ranked 1
        avg_df[RANK_COL] = avg_df.groupby(DATASET_COL)[AUC_COL].rank(ascending=False, method='first')
        ranked_df = avg_df.sort_values(by=[DATASET_COL, AUC_COL], ascending=[True, False])

        export_path = self.version_path / "ranked_results_filtered.csv"
        ranked_df.to_csv(export_path, index=False)
        df.to_csv(self.version_path / "non_aggregated.csv", index=False)
        self._create_rank_plot(ranked_df, group_by=BASE_COL)

        ranked_df[METHOD_COL] = ranked_df[METHOD_COL].apply(self.rename)
        ranked_df[SPACE_COL] = ranked_df[SPACE_COL].replace("Embedding", "Avg").replace("Token-space", "Avg").replace("Word NPTE", "NPTE")
        ranked_df.to_csv(self.version_path / "ranked_results_filterd_renamed.csv", index=False)

        self._create_rank_plot(ranked_df, group_by=[BASE_COL, SPACE_COL] , name="ranked_renamed", method_color=coloring)
        return
        rank = RankVisualizer([], self.version_path)
        rank.create_box_plot_vertical(data=ranked_df,
                                      method_col=METHOD_COL,
                                      metric_col=AUC_COL,
                                      group_by=[BASE_COL, SPACE_COL],
                                      method_color=coloring, name="ranked_renamed_vertical")

    def rename(self, method_name: str) -> str:
        #if method_name.endswith("+ Embedding"):
        method_name = method_name.replace("+ Embedding", AVG_SPACE)

        method_name = method_name.replace("+ Word NPTE", "NPTE")

        if method_name.startswith("FeatureBagging"):
            method_name = FEATURE_BAGGING# + " " + method_name.split(" ")[-1]
            return method_name

        if method_name.startswith("VMMD"):# and "E" in method_name:
            return V_GAN

        #if method_name.startswith("VMMD") and "W" in method_name:
            #return V_GAN_NPTE

        if method_name.startswith("TokenVMMD"):
            return V_GAN_TOKEN

        if method_name.startswith("TextVMMD"):
            return V_GAN_TEXT

        if method_name == TRIVIAL:
            return TRIVIAL

        return FULL_SPACE# + " " + method_name.split(" ")[-1]


    def _create_rank_plot(self, ranked_df, group_by: str or list, name:Optional[str] = None, method_color: Optional[Callable[[str], str]] = None):
        """
        Creates a box plot of the ranks for each method and metric, grouped by base method.
        """

        rank_visualizer = RankVisualizer([], self.version_path)
        rank_visualizer.create_box_plot(data=ranked_df,
                                        method_col=METHOD_COL,
                                        metric_col=AUC_COL,
                                        group_by=group_by, name=name, method_color=method_color)


if __name__ == "__main__":
    # Example usage
    version_path = Path(os.path.dirname(__file__)) / '..' / '..' / 'results' / 'outlier_detection' / '0.45'
    aggregator = ResultAggregator(version_path)
    aggregator.aggregate()