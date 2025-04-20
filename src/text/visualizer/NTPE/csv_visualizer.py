from pathlib import Path

import pandas as pd
from pandas import DataFrame

from text.consts.columns import TYPE_COL, TRAIN_SIZE_COL, TEST_SIZE_COL, EMB_MODEL_COL, PROMPT_COL, RUN_COL
from text.outlier_detection.odm import METHOD_COL, SPACE_COL, AUC_COL, PRAUC_COL, TIME_TAKEN_COL, BASE_COL, DATASET_COL, \
    F1_COL


class CsvVisualizer():
    """
    A visualizer for the CSV files. It takes the experiment data from the
    embedding comparisons between the different aggregation methods
    and creates a CSV file with the results, that is more presentable.
    """
    def __init__(self, export_path: Path, experiment_data: DataFrame):
        self.export_path = export_path # Includes file name
        self.experiment_data = experiment_data
        expected_columns = [METHOD_COL, SPACE_COL, AUC_COL,
                            PRAUC_COL, TIME_TAKEN_COL, BASE_COL,
                            TYPE_COL, DATASET_COL, TRAIN_SIZE_COL,
                            TEST_SIZE_COL, EMB_MODEL_COL, PROMPT_COL,
                            RUN_COL]
        for col in expected_columns:
            if col not in self.experiment_data.columns:
                raise ValueError(f"Missing column {col} in experiment data")

    def export_csv(self):
        """
        Exports the experiment data to a CSV file.
        """

        filled = self.experiment_data.copy()
        filled[PROMPT_COL] =   filled[PROMPT_COL].fillna("None")

        pivoted = filled.pivot_table(
            index=[DATASET_COL, PROMPT_COL],
            columns=[EMB_MODEL_COL, BASE_COL, TYPE_COL],
            values=AUC_COL,
            aggfunc='mean'
        )

        # Flatten the multi-index columns for better readability
        pivoted.columns = ['_'.join(map(str, col)).strip() for col in pivoted.columns]

        pivoted.reset_index(inplace=True)

        avg_rows = pivoted[pivoted[PROMPT_COL] == "None"].copy().reset_index(drop=True).drop(columns=[PROMPT_COL])
        ntpe_rows = pivoted[pivoted[PROMPT_COL] != "None"].copy().reset_index(drop=True)

        # Fill the missing values in the columns with the avg type in the ntpe rows df with the values in the avg rows.
        for col in avg_rows.columns:
            if col not in [DATASET_COL, PROMPT_COL]:
                for index, row in ntpe_rows.iterrows():
                    if pd.isna(row[col]):
                        dataset = row[DATASET_COL]
                        avg_value= avg_rows[avg_rows[DATASET_COL] == dataset][col]
                        assert len(avg_value) == 1, f"Multiple values found for dataset {dataset} in column {col}"
                        ntpe_rows.at[index, col] = avg_value.iloc[0]
        ntpe_rows[PROMPT_COL] = ntpe_rows[PROMPT_COL].apply(lambda x: x.replace(",", "[comma]")).apply(lambda x: x.replace("\n", "\\n"))


        ntpe_rows.to_csv(self.export_path , index=False)