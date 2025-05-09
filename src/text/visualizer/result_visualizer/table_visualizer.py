from pathlib import Path

import pandas as pd

from text.outlier_detection.odm import DATASET_COL, AUC_COL, TIME_TAKEN_COL, BASE_COL
from text.outlier_detection.pyod_odm import ECOD


def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def generate_ieeetran_table(df, caption, label):
    cols = df.columns.tolist()
    # Truncate all float values to 2 decimal places
    for i in range(len(cols)):
        if df[cols[i]].dtype == 'float64':
            df[cols[i]] = df[cols[i]].apply(lambda x: truncate(x, 4))
    # alignment: first column left, rest centered
    align = 'l' + 'c' * (len(cols) - 1)
    lines = []
    lines.append(r'\begin{table*}[!t]')
    lines.append(r'\centering')
    lines.append(r'\caption{' + caption + r'}')
    lines.append(r'\label{' + label + r'}')
    lines.append(r'\begin{tabular}{' + align + r'}')
    lines.append(r'\toprule')
    # header
    header = ' & '.join(cols) + r' \\'
    lines.append(header)
    lines.append(r'\midrule')
    # rows
    for _, row in df.iterrows():
        vals = []
        for v in row:
            s = f"{v}"
            # escape underscores
            s = s.replace('_', r'\_')
            vals.append(s)
        lines.append(' & '.join(vals) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')
    code = '\n'.join(lines)
    #code = code.replace("_", r"\_")
    return code

if __name__ == "__main__":
    version_path = Path(__file__).parent.parent.parent.parent.parent / "experiments" / "0.45"
    ranked_df = pd.read_csv(version_path / "ranked_results_filterd_renamed.csv")


    # Set dtype of the time column to int
    ranked_df[TIME_TAKEN_COL] = ranked_df[TIME_TAKEN_COL].astype(int)

    ranked_df.columns = [col.replace("time_taken", "time_taken (s)") for col in ranked_df.columns]
    ranked_df.columns = [col.replace("_", " ") for col in ranked_df.columns]
    ranked_df = ranked_df[ranked_df[BASE_COL] != "ECOD"]

    #Remove last column
    ranked_df = ranked_df.iloc[:, :-1]
    # Split the dataset by each dataset

    datasets = ranked_df[DATASET_COL].unique()
    chunks = []
    for dataset in datasets:
        chunk = ranked_df[ranked_df[DATASET_COL] == dataset]
        # Remove the dataset column
        chunk = chunk.drop(columns=[DATASET_COL])
        # Sort by AUC
        chunk = chunk.sort_values(by=[AUC_COL], ascending=False)
        dataset = dataset.replace("_", " ")
        chunks.append((dataset, chunk))

    # Generate LaTeX tables for each chunk
    for dataset, chunk in chunks:
        caption = f"Ranked results for {dataset}."
        label = f"tab:ranked_results_{dataset}"
        latex_table = generate_ieeetran_table(chunk, caption, label)
        print("\n\n\n")
        print(latex_table)
        #with open(version_path / f"ranked_results_chunk_{i+1}.tex", 'w') as f:
            #f.write(latex_table)
            #f.write("\n\n%--------------------------------------------------------------\n\n")