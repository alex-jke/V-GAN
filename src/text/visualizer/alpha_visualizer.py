import os
from pathlib import Path

import torch
from matplotlib import pyplot as plt

from .visualizer import Visualizer

SUBSPACE_COLUMN = 'subspace'
SUBSPACE_PROBABILITY_COLUMN = 'probability'


class AlphaVisualizer(Visualizer):
    """
    Class for visualizing data with using the alpha in text.
    """

    def visualize(self, samples: int = 1):
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
        output_path = self.output_dir / "alpha_visualization.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Visualization saved to {output_path}. Open this file in a web browser to view.")
