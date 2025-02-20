from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


class ResultVisualizer():

    def __init__(self, result: pd.DataFrame, output_dir: Path):
        self.result = result
        self.output_dir = output_dir
        self.bar = "bar"
        self.line = "line"

    def visualize(self, x_column: str, y_column: str, type="bar"):
        x_values = self.result[x_column]
        y_values = self.result[y_column]
        try:
            x_values, y_values = zip(*sorted(zip(x_values, y_values)))
        except TypeError as e:
            print(e)
            raise e
        plt.figure(figsize=(12, 8))  # Set the figure size
        if type == self.bar:
            plt.barh(x_values, y_values)
            x_column, y_column = y_column, x_column
        else:
            plt.plot(x_values, y_values, 'o-', color='blue')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        #plt.xticks(rotation=90)
        plt.title(f"{y_column} by {x_column}")
        plt.tight_layout()
        path = self.output_dir / f"{y_column}_by_{x_column}.png"
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
