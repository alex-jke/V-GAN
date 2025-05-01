from pathlib import Path
import os
import pandas as pd
from text.consts.columns import EMB_MODEL_COL, DATASET_COL
from text.outlier_detection.odm import METHOD_COL, SPACE_COL, AUC_COL, PRAUC_COL, F1_COL, TIME_TAKEN_COL, BASE_COL

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

        # Average over the runs
        avg_df = df.groupby(group_by_columns).mean().reset_index()

        # Group by dataset and create rankings from the average AUC, where the best AUC is ranked 1
        avg_df["rank"] = avg_df.groupby(DATASET_COL)[AUC_COL].rank(ascending=False, method='first')

        export_path = self.version_path / "ranked_results.csv"
        avg_df.to_csv(export_path, index=False)


if __name__ == "__main__":
    # Example usage
    version_path = Path(os.path.dirname(__file__)) / '..' / '..' / 'results' / 'resuts_remote' / '0.44'
    aggregator = ResultAggregator(version_path)
    aggregator.aggregate()