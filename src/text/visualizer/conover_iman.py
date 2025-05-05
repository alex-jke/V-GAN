import os
from pathlib import Path

import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import numpy as np

from text.consts.columns import RANK_COL
from text.outlier_detection.odm import METHOD_COL
from text.visualizer.result_visualizer.result_aggregator import V_GAN_AVG, V_GAN_NPTE, V_GAN_TEXT, V_GAN_TOKEN, \
    FULL_SPACE

# Specify the odm
odm = "CBLOF"


def conover_iman_table_generator(odm, u_type):
    # Load the data
    #df = pd.read_csv(f'experiments/Outlier_Detection/COMPETITORS/full_tables_with_rank/{odm}_res.csv', delimiter=',')
    visualizer_path = Path(os.path.dirname(__file__))
    df_path = visualizer_path.parent / "results" / "outlier_detection" / "0.45" / "ranked_results_filterd_renamed.csv"
    df = pd.read_csv(df_path)

    methods = [V_GAN_AVG, V_GAN_NPTE, V_GAN_TEXT, V_GAN_TOKEN, FULL_SPACE]

    ranks_vgan_avg = df[RANK_COL][df[METHOD_COL] == "V-GAN-Avg"]

    # Using a lense operator
    if u_type == "lense":
        df = df[df['p-value'] > 0.10]
    else:
        df = df[df['p-value'] < 0.10]
    # Extracting the ranks for each model
    ranks_cae = df['RANK CAE'].dropna()
    ranks_hics = df['RANK HiCS'].dropna()
    ranks_vgan = df['RANK VGAN'].dropna()
    ranks_clique = df['RANK CLIQUE'].dropna()
    ranks_elm = df['RANK ELM'].dropna()
    ranks_gmd = df['RANK GMD'].dropna()
    ranks_pca = df['RANK PCA'].dropna()
    ranks_umap = df['RANK UMAP'].dropna()


    # Performing the Kruskal-Wallis test
    kruskal_result = stats.kruskal(ranks_cae, ranks_hics, ranks_clique, ranks_elm, ranks_gmd, ranks_pca, ranks_umap,
                                   ranks_vgan)
    print(f"Kruskal-Wallis Test Result: {kruskal_result}")

    # If the p-value from Kruskal-Wallis test is significant, perform Conover-Iman test
    conover_result = sp.posthoc_conover(
        [ranks_cae, ranks_hics, ranks_clique, ranks_elm, ranks_gmd, ranks_pca, ranks_umap, ranks_vgan])
    conover_result.index = ["CAE", "HiCS", "CLIQUE", "ELM", "GMD", "PCA", "UMAP", "VGAN"]
    conover_result.columns = ["CAE", "HiCS", "CLIQUE", "ELM", "GMD", "PCA", "UMAP", "VGAN"]

    # Calculate the average ranks for each method
    avg_ranks = pd.DataFrame({
        'CAE': [np.mean(ranks_cae)],
        'HiCS': [np.mean(ranks_hics)],
        'CLIQUE': [np.mean(ranks_clique)],
        'ELM': [np.mean(ranks_elm)],
        'GMD': [np.mean(ranks_gmd)],
        'PCA': [np.mean(ranks_pca)],
        'UMAP': [np.mean(ranks_umap)],
        'VGAN': [np.mean(ranks_vgan)]
    }).T
    avg_ranks.columns = ['Average Rank']
    avg_ranks = avg_ranks.sort_values(by='Average Rank')  # Sort by rank if necessary

    # Create a matrix for storing the + and ++ symbols
    symbol_matrix = pd.DataFrame('', index=conover_result.index, columns=conover_result.columns)

    # Fill in the matrix based on the p-value and average rank conditions
    for row in conover_result.index:
        for col in conover_result.columns:
            if row != col:  # No comparison with itself
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
    symbol_matrix.to_csv(f"experiments/Outlier_Detection/COMPETITORS/conover-iman/{odm}_{u_type}.csv")
    return symbol_matrix, conover_result

if __name__ == "__main__":
    conover_iman_table_generator("LOF", "lense")
