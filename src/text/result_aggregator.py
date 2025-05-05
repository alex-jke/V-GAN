import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd


class FullResultAggregator():
    """
    A class to aggregate the results of multiple experiments. The results are stored in a nested folder structure, where each folder contains the results of one experiment.
    This class traverses the folder structure and reads the results of each experiment. The results are stored in a list of dataframes, where each dataframe contains two cross joined dataframes: one with the common metrics and one with the results of the experiment.
    It then provides methods to aggregate the results and save them to a csv file. One such aggregation is for each "method" in the results dataframe to find the best result for each metric: auc, prauc, f1. That is, create one dataframe with the best result for each metric for each method.
    It then stores the aggregated results in a csv file in the results folder, under the name "aggregated_results.csv".
    """

    def __init__(self, common_metrics_name: str, result_name: str):
        self.results_path = Path(os.path.dirname(__file__)) / "results"
        self.common_metrics_name = common_metrics_name
        self.results_name = result_name
        self.exist_criteria_file = self.common_metrics_name
        self.results: List[pd.DataFrame] = []

    def is_experiment_result_dir(self, path: Path) -> bool:
        """
        Check if a given path is a directory containing the results of an experiment.
        It does so by verifying that the directory contains the criteria file.
        """
        if not path.is_dir():
            return False
        criteria_file = path / self.exist_criteria_file
        return criteria_file.exists()

    def traverse_and_load_results(self):
        """
        Traverse the nested folder structure starting from self.results_path,
        find all experiment result directories, and load their results into self.results.
        Each experiment folder is expected to contain two files:
          - one with the common metrics (self.common_metrics_name)
          - one with the experiment results (self.results_name)
        """
        folders = self.results_path.glob('**/*')
        error_occurred = False
        for folder in folders:
            if folder.is_dir() and self.is_experiment_result_dir(folder):
                try:
                    common_metrics_file = folder / self.common_metrics_name
                    results_file = folder / self.results_name

                    common_df = pd.read_csv(common_metrics_file)
                    results_df = pd.read_csv(results_file)

                    df = results_df.join(common_df, how='cross')

                    self.results.append(df)
                    #print(f"Loaded results from: {folder}")
                except Exception as e:
                    if not error_occurred:
                        print(f"Failed to load results from {folder}: {e}", end=" ")
                    else:
                        print(f"and {folder}", end=" ")

    def aggregate_best_results(self) -> pd.DataFrame:
        """
        Aggregate the best results for each method across all experiments.
        For each method in the results dataframe, find the best (i.e. maximum) value for each metric: auc, prauc, f1.
        Returns a dataframe with aggregated best results.
        """
        # Combine all experiment results into a single DataFrame.
        results_list = [results_df for results_df in self.results]
        if not results_list:
            print("No experiment results found to aggregate.")
            return pd.DataFrame()

        combined_df = pd.concat(results_list, ignore_index=True)

        # Ensure that the necessary columns exist.
        required_columns = ['method', 'auc', 'prauc', 'f1']
        for col in required_columns:
            if col not in combined_df.columns:
                raise ValueError(f"Column '{col}' not found in the results data.")

        best_rows = []
        # For each metric, find the row for each method that has the maximum value.
        for metric in ['auc', 'prauc', 'f1']:
            # idxmax returns the index of the max value for each group.
            idx = combined_df.groupby("dataset")[metric].idxmax()
            best_df = combined_df.loc[idx].copy()
            best_df['best_metric'] = metric
            #best_df = pd.DataFrame([best_df])
            best_rows.append(best_df)

        # Concatenate the best rows for all metrics.
        aggregated_df = pd.concat(best_rows, ignore_index=True)
        return aggregated_df

    def save_aggregated_results(self, aggregated_df: pd.DataFrame):
        """
        Save the aggregated results to a CSV file in the results folder.
        The file is named 'aggregated_results.csv'.
        """
        output_file = self.results_path / "aggregated_results.csv"
        aggregated_df.to_csv(output_file, index=False)
        print(f"Aggregated results saved to: {output_file}")

    def run_aggregation(self):
        """
        Execute the complete aggregation process:
          1. Traverse the folder structure and load results.
          2. Aggregate the best results for each method.
          3. Save the aggregated results to a CSV file.
        """
        self.traverse_and_load_results()
        aggregated_df = self.aggregate_best_results()
        if not aggregated_df.empty:
            self.save_aggregated_results(aggregated_df)
        else:
            print("No aggregated results to save.")

if __name__ == "__main__":
    aggregator = FullResultAggregator(result_name="results.csv",
                                  common_metrics_name="comon_metrics.csv")
    aggregator.run_aggregation()
