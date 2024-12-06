import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from ..Embedding.tokenizer import Tokenizer
from .visualizer import Visualizer

SUBSPACE_COLUMN = 'subspace'
SUBSPACE_PROBABILITY_COLUMN = 'probability'


class AlphaVisualizer(Visualizer):
    """
    Class for visualizing data with using the alpha in text.
    """

    def __init__(self, model, tokenized_data: Tensor, tokenizer: Tokenizer):  # todo: replace subspaces with model.
        super().__init__()
        self.num_subspaces = 500
        # Tensor is of the shape (num_subspaces, sequence_length)
        subspaces: Tensor = model.generate_subspaces(self.num_subspaces)
        # unique_subspaces, proba = np.unique(np.array(subspaces.to('cpu')), axis=0, return_counts=True)
        # proba = proba / np.array(subspaces.to('cpu')).shape[0]

        # unique_subspaces = [Tensor([int(boolean) * proba[i] for boolean in boolean_list]) for i, boolean_list in enumerate(unique_subspaces.tolist())]

        # self.subspace_df: pd.DataFrame = pd.DataFrame({SUBSPACE_COLUMN: unique_subspaces, SUBSPACE_PROBABILITY_COLUMN: proba})
        self.avg_subspace: Tensor = subspaces.sum(dim=0) / self.num_subspaces

        self.tokenized_data = tokenized_data
        self.tokenizer: Tokenizer = tokenizer

    def visualize(self, samples: int = 1):
        """
        Visualizes the data with alpha characters.
        :param samples: The number of samples to visualize.
        """
        sample_data = self.tokenized_data[:samples]  # todo: get random samples
        most_selected = torch.argsort(self.avg_subspace, descending=True)[:10]
        padding_token = self.tokenizer.detokenize([self.tokenizer.padding_token])
        print("indices and value of most selected features:",
              [(int(i), float(self.avg_subspace[i])) for i in most_selected])
        print("average subspace:", self.avg_subspace)
        # fig, ax = plt.subplots()
        x_dim = 15
        y_dim = 20
        spacing = 0.0004
        fig = plt.figure(figsize=(x_dim, y_dim))  # todo: create a html file instead of a pdf, to make it scrollable
        y_pos = 1
        for i in range(samples):
            print("Sample", i)
            int_list = [int(number) for number in sample_data[i].tolist()]
            print("Original:", self.tokenizer.detokenize(int_list))
            strings = [self.tokenizer.detokenize([token]) for token in int_list if token != self.tokenizer.padding_token]

            # Normalize the average subspace values
            values = self.avg_subspace[:len(strings)]
            values = (values - values.min()) / (values.max() - values.min()) # (self.avg_subspace - self.avg_subspace.min()) / (self.avg_subspace.max() - self.avg_subspace.min())
            # Create a figure and axis

            # Loop through strings and their transparency values
            x_pos = 0.01  # Starting x position
            y_pos -= y_dim / 800  # Adjusted vertical spacing
            for j, (string, alpha) in enumerate(zip(strings, values)):
                if False or string == padding_token:
                    string = "-"
                if x_pos > 0.95:
                    y_pos -= y_dim / 2000  # Adjusted vertical spacing
                    x_pos = 0.01
                fig.text(
                    x_pos,
                    y_pos,  # Adjusted vertical spacing
                    string,
                    fontsize=12,
                    ha='left',
                    va='center',
                    family='monospace',  # Fixed-width font for better spacing
                    # color=(0, 0, 0, float(alpha))  # RGBA values (alpha is transparency)
                    color=(float(alpha), 0, 0, 1)
                )

                x_pos += x_dim * spacing * len(string) + x_dim * spacing

            # Hide axes
            plt.axis('off')

        # Show the plot
        plt.tight_layout()
        path = Path(os.getcwd()) / "text" / "results" / "alpha_visualization.pdf"
        plt.savefig(path)
        plt.show()

    def visualize_html(self, samples: int = 1):
        """
        Visualizes the data with alpha characters.
        :param samples: The number of samples to visualize.
        """
        sample_data = self.tokenized_data[:samples]  # Select the first 'samples' data points
        most_selected = torch.argsort(self.avg_subspace, descending=True)[:10]
        padding_token = self.tokenizer.detokenize([self.tokenizer.padding_token])

        print("indices and value of most selected features:",
              [(int(i), float(self.avg_subspace[i])) for i in most_selected])
        print("average subspace:", self.avg_subspace)

        # Initialize HTML content
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Alpha Visualization</title>
            <style>
                body {
                    font-family: monospace;
                    line-height: 1.5;
                    margin: 20px;
                    white-space: pre-wrap;
                }
                .token {
                    display: inline-block;
                    margin-right: 5px;
                }
            </style>
        </head>
        <body>
        """

        y_pos = 1  # Vertical position
        for i in range(samples):
            print("Sample", i)
            int_list = [int(number) for number in sample_data[i].tolist()]
            print("Original:", self.tokenizer.detokenize(int_list))
            strings = [self.tokenizer.detokenize([token]) for token in int_list if
                       token != self.tokenizer.padding_token]

            # Normalize the average subspace values
            values = self.avg_subspace[:len(strings)]
            values = (values - values.min()) / (values.max() - values.min())

            html_content += f"<div><strong>Sample {i + 1}:</strong></div>"

            # Loop through strings and their transparency values
            for string, alpha in zip(strings, values):
                if string == padding_token:
                    string = "-"
                color_intensity = int(alpha * 255)
                html_content += f'<span class="token" style="color: rgba({color_intensity}, 0, 0, 1);">{string}</span>'
            html_content += "<br><br>"

        # Close the HTML content
        html_content += """
        </body>
        </html>
        """

        # Save HTML content to a file
        output_path = Path(os.getcwd()) / "text" / "results" / "alpha_visualization.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Visualization saved to {output_path}. Open this file in a web browser to view.")
