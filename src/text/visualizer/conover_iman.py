import os
from pathlib import Path
from typing import Optional, List

import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import numpy as np

from text.consts.columns import RANK_COL, DATASET_COL
from text.dataset.nlp_adbench import NLP_ADBench
from text.outlier_detection.odm import METHOD_COL, SPACE_COL, BASE_COL
from text.visualizer.result_visualizer.result_aggregator import V_GAN, V_GAN_TEXT, V_GAN_TOKEN, \
    FULL_SPACE, FEATURE_BAGGING

# Specify the odm
fully_myopic_datasets = [NLP_ADBench.sms_spam().name, NLP_ADBench.n24news().name, NLP_ADBench.yelp_review_polarity().name]


def conover_iman_table_generator(odm, u_type, df: pd.DataFrame, space: str, export_path: Optional[Path]=None):
    # Load the data
    #df = pd.read_csv(f'experiments/Outlier_Detection/COMPETITORS/full_tables_with_rank/{odm}_res.csv', delimiter=',')s

    if u_type == "lense":
        df = df[df[DATASET_COL].isin(fully_myopic_datasets)]
    elif u_type == "all":
        df = df
    else:
        df = df[~df[DATASET_COL].isin(fully_myopic_datasets)]

    df = df[df[SPACE_COL] == space]
    df = df[df[BASE_COL] == odm]

    methods = [V_GAN, V_GAN_TEXT, V_GAN_TOKEN, FULL_SPACE, FEATURE_BAGGING, "Trivial"]

    methods = [method for method in methods if method in df[METHOD_COL].values]

    #ranks_vgan_avg = df[RANK_COL][df[METHOD_COL] == "V-GAN-Avg"]
    ranks = {method: df[RANK_COL][df[METHOD_COL] == method] for method in methods}

    kruskal_result = stats.kruskal(*ranks.values())
    print(f"Kruskal-Wallis Test Result: {kruskal_result}")

    # Perform the Conover-Iman test
    conover_result = sp.posthoc_conover(list(ranks.values()))
    conover_result.index = methods
    conover_result.columns = methods

    # Calculate the average ranks for each method
    avg_ranks = pd.DataFrame({
        method: [np.mean(ranks[method])] for method in methods
    }).T
    avg_ranks.columns = ['Average Rank']
    avg_ranks = avg_ranks.sort_values(by='Average Rank')  # Sort by rank if necessary

    # Create a matrix for storing the + and ++ symbols
    symbol_matrix = pd.DataFrame('', index=conover_result.index, columns=conover_result.columns)

    # Fill in the matrix based on the p-value and average rank conditions
    for row in conover_result.index:
        for col in conover_result.columns:
            if row != col:
                p_value = conover_result.loc[row, col]
                if p_value <= 0.10 and avg_ranks.loc[row, 'Average Rank'] < avg_ranks.loc[col, 'Average Rank']:
                    if p_value <= 0.05:
                        symbol_matrix.loc[row, col] = '++'
                    else:
                        symbol_matrix.loc[row, col] = '+'
                if p_value <= 0.10 and avg_ranks.loc[row, 'Average Rank'] > avg_ranks.loc[col, 'Average Rank']:
                    if p_value <= 0.05:
                        symbol_matrix.loc[row, col] = '--'
                    else:
                        symbol_matrix.loc[row, col] = '-'
            if row == col:
                symbol_matrix.loc[row, col] = "="

    print(f"\nThis leads to a symbolic matrix of:\n {symbol_matrix}")
    #symbol_matrix.to_csv(f"experiments/Outlier_Detection/COMPETITORS/conover-iman/{odm}_{u_type}.csv")
    symbol_matrix.to_csv(export_path / f"symbol_matrix_{odm}.csv")
    return symbol_matrix, conover_result


    # Using a lense operator
    if u_type == "lense":
        df = df[df['p-value'] > 0.10]
    else:
        df = df[df['p-value'] < 0.10]


def symbol_matrices_to_latex(symbol_matrices: List[pd.DataFrame], odms: List[str]) -> str:
    """
    Convert the symbol matrices to a LaTeX table format that spans both columns in IEEE format.
    The table will be positioned at the top of the page.
    """
    if not symbol_matrices or len(symbol_matrices) != len(odms):
        return "Error: Symbol matrices and ODMs lists must be non-empty and have the same length"

    # Get the methods (rows) from the first matrix
    methods = symbol_matrices[0].index.tolist()

    # Start building the LaTeX table with table* environment for double column
    latex = "\\begin{table*}[t]\n"  # t for top placement
    latex += "\\centering\n"
    latex += "\\footnotesize\n"  # Reduce font size to fit wide table
    latex += "\\caption{Conover-Iman test results}\n"

    # Create column specification
    col_count = sum(len(matrix.columns) for matrix in symbol_matrices)
    latex += "\\begin{tabular}{l" + "|" + "c" * col_count + "}\n"
    latex += "\\hline\n"

    # Add the header with ODM names spanning their respective columns
    latex += " & "
    for i, odm in enumerate(odms):
        cols = len(symbol_matrices[i].columns)
        latex += f"\\multicolumn{{{cols}}}{{c|}}{{{odm}}}" if i < len(odms) - 1 else f"\\multicolumn{{{cols}}}{{c}}{{{odm}}}"
        if i < len(odms) - 1:
            latex += " & "
    latex += " \\\\\n\\hline\n"

    # Add the column headers (method names)
    latex += "Method & "
    for i, matrix in enumerate(symbol_matrices):
        cols = matrix.columns.tolist()
        latex += " & ".join(cols)
        if i < len(symbol_matrices) - 1:
            latex += " & "
    latex += " \\\\\n\\hline\n"

    # Add the data rows
    for method in methods:
        latex += f"{method} & "
        for i, matrix in enumerate(symbol_matrices):
            cols = matrix.columns.tolist()
            row_values = [str(matrix.loc[method, col]) for col in cols]
            latex += " & ".join(row_values)
            if i < len(symbol_matrices) - 1:
                latex += " & "
        latex += " \\\\\n"

    # Close the table
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\label{tab:conover_iman_results}\n"
    latex += "\\end{table*}\n"

    return latex


if __name__ == "__main__":
    version_path = Path(__file__).parent.parent.parent.parent / "experiments" / "0.45"
    df_path = version_path / "ranked_results_filterd_renamed.csv"
    ranked_results = pd.read_csv(df_path)
    symbol_matrices: list = []
    results = []
    space = "Avg"
    odms = ["LUNAR", "LOF"]
    for odm in odms:
    #odm = "LUNAR"
        ranked_results = ranked_results[ranked_results[DATASET_COL] != NLP_ADBench.n24news().name]

        symbol_matrix, conover_results = conover_iman_table_generator(odm, "all", ranked_results, space, export_path=version_path)
        symbol_matrices.append(symbol_matrix)
        results.append(conover_results)
    #concated_matrix = concatenate_symbol_matrices(symbol_matrices, odms)
    latex_table = symbol_matrices_to_latex(symbol_matrices, odms)
    print(latex_table)
