import os
from pathlib import Path
from typing import Optional, List

import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import numpy as np

from text.consts.columns import RANK_COL, DATASET_COL
from text.dataset.nlp_adbench import NLP_ADBench
from text.outlier_detection.odm import METHOD_COL, SPACE_COL, BASE_COL, AUC_COL
from text.visualizer.result_visualizer.result_aggregator import V_GAN, V_GAN_TEXT, V_GAN_TOKEN, \
    FULL_SPACE, FEATURE_BAGGING, AVG_SPACE

# Specify the odm
fully_myopic_datasets_NPTE = [NLP_ADBench.sms_spam().name,
                              NLP_ADBench.n24news().name,
                              NLP_ADBench.yelp_review_polarity().name,
                              #NLP_ADBench.agnews().name,
                              #NLP_ADBench.emotion().name
                              ]
fully_myopic_datasets_avg = [NLP_ADBench.sms_spam().name,
                              NLP_ADBench.n24news().name,
                              NLP_ADBench.yelp_review_polarity().name,
                              #NLP_ADBench.movie_review().name
                             ]
all_datasets = [ds.name for ds in NLP_ADBench.get_all_datasets()]


def conover_iman_table_generator(odm, u_type, df: pd.DataFrame, space: str, export_path: Optional[Path]=None):
    # Load the data
    #df = pd.read_csv(f'experiments/Outlier_Detection/COMPETITORS/full_tables_with_rank/{odm}_res.csv', delimiter=',')s
    fully_myopic_datasets = fully_myopic_datasets_NPTE if space == "NPTE" else fully_myopic_datasets_avg

    if u_type == "lense":
        df = df[df[DATASET_COL].isin(fully_myopic_datasets)]
    elif u_type == "all":
        df = df
    else:
        df = df[~df[DATASET_COL].isin(fully_myopic_datasets)]

    df = df[df[SPACE_COL] == space]
    df = df[df[BASE_COL] == odm]

    methods = [V_GAN, V_GAN_TEXT, V_GAN_TOKEN, FULL_SPACE, FEATURE_BAGGING]#, "Trivial"]

    methods = [method for method in methods if method in df[METHOD_COL].values]

    #ranks_vgan_avg = df[RANK_COL][df[METHOD_COL] == "V-GAN-Avg"]
    #ranks = {}
    #Rerank for the given group
    #for method in methods:
    df[RANK_COL] = df.groupby(DATASET_COL)[AUC_COL].rank(ascending=False, method='min')

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

    #conover_result.columns = [col if col != FEATURE_BAGGING else "FB" for col in conover_result.columns]

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
                        symbol_matrix.loc[row, col] = '- -'
                    else:
                        symbol_matrix.loc[row, col] = '-'
            if row == col:
                symbol_matrix.loc[row, col] = "="

    print(f"\nThis leads to a symbolic matrix of:\n {symbol_matrix}")
    #symbol_matrix.to_csv(f"experiments/Outlier_Detection/COMPETITORS/conover-iman/{odm}_{u_type}.csv")
    symbol_matrix.columns = [col if col != FEATURE_BAGGING else "FB" for col in symbol_matrix.columns]
    first_column = symbol_matrix.columns[0]
    #symbol_matrix[first_column] = symbol_matrix[first_column].apply(lambda method: method if method != FEATURE_BAGGING else FEATURE_BAGGING + " (FB)")
    #symbol_matrix.to_csv(export_path / f"symbol_matrix_{odm}.csv")
    return symbol_matrix, conover_result


    # Using a lense operator
    if u_type == "lense":
        df = df[df['p-value'] > 0.10]
    else:
        df = df[df['p-value'] < 0.10]


def symbol_matrices_to_latex(symbol_matrices: List[pd.DataFrame],
                             odms: List[str],
                             space: str,
                             u_type: str,
                             datasets: List[str]) -> str:
    blocks = [m.columns.tolist() for m in symbol_matrices]
    methods = symbol_matrices[0].index.tolist()

    # 1) table header
    col_spec = "l" + "".join(f"|{'c'*len(b)}" for b in blocks) + "|"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{Conover–Iman test results. Space: {space}, type: {u_type}, datasets: {', '.join(datasets)}}}",
        rf"\label{{tab:conover_iman_{space}_{u_type}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        # First header row: ODM names
        " & " + " & ".join(
            rf"\multicolumn{{{len(blocks[i])}}}{{c}}{{\textbf{{{odm}}}}}"
            for i, odm in enumerate(odms)
        ) + r" \\ \hline",
        # Second header row: method names
        " & " + " & ".join(
            " & ".join(rf"\textbf{{{col}}}" for col in blocks[i])
            for i in range(len(blocks))
        ) + r" \\ \hline",
    ]

    # 2) data rows with grey‐shading for row_idx == col_idx
    for ri, method in enumerate(methods):
        row_cells = []
        for bi, cols in enumerate(blocks):
            for ci, col in enumerate(cols):
                if ri == ci:
                    row_cells.append(r"\cellcolor{gray!25}{}")
                else:
                    row_cells.append(symbol_matrices[bi].loc[method, col])
        lines.append(f"{method} & " + " & ".join(row_cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]

    return "\n".join(lines)


if __name__ == "__main__":
    version_path = Path(__file__).parent.parent.parent.parent / "experiments" / "0.45" / "rank_csv"
    df_path = version_path / "ranked_results_filterd_renamed.csv"
    ranked_results = pd.read_csv(df_path)

    #space = "Avg"
    #u_type = "lense"
    spaces = ["Avg",
              "NPTE"
              ]
    u_types = [#"lense",
               #"non-lense"
                "all",
               ]
    odms = ["LUNAR", "LOF"]
    latex_tables = []
    for space in spaces:
        for u_type in u_types:
            symbol_matrices: list = []
            results = []
            for odm in odms:
            #odm = "LUNAR"
                #ranked_results = ranked_results[ranked_results[DATASET_COL] != NLP_ADBench.n24news().name]

                symbol_matrix, conover_results = conover_iman_table_generator(odm, u_type, ranked_results, space, export_path=version_path)
                symbol_matrices.append(symbol_matrix)
                results.append(conover_results)
            #concated_matrix = concatenate_symbol_matrices(symbol_matrices, odms)
            latex_table = symbol_matrices_to_latex(symbol_matrices, odms, space=space, u_type=u_type, datasets=odms)
            latex_tables.append(latex_table)
    print("----------------------\n")
    seperator = "\n\n%---------------------------------------------\n\n"
    print(seperator.join(latex_tables))
