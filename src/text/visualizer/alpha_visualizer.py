import os
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch import Tensor

from .visualizer import Visualizer

SUBSPACE_COLUMN = 'subspace'
SUBSPACE_PROBABILITY_COLUMN = 'probability'


class AlphaVisualizer(Visualizer):
    """
    Class for visualizing data with using the alpha in text.
    """

    def export_html(self, sample_data: Tensor, subspaces: Tensor, folder_appendix: str, epoch: int = -1):
        padding_token = self.tokenizer.detokenize([self.tokenizer.padding_token])

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
        for i in range(sample_data.size(0)):
            # print("Sample", i)
            int_list = [int(number) for number in sample_data[i].tolist()]
            # print("Original:", self.tokenizer.detokenize(int_list))
            strings = [self.tokenizer.detokenize(token) for token in int_list if
                       token != self.tokenizer.padding_token]
            sample_length = len(strings)

            # Normalize the average subspace values
            values = subspaces[i][:sample_length]
            max, min = values.max(), values.min()
            if max != min:
                values = (values - min) / (max - min)
                # print("Normalized values:", values)

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
        postfix = f"_{epoch}" if epoch >= 0 else ""
        output_path = self.output_dir / f"text_{folder_appendix}" / f"text{postfix}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        #print(f"Visualization saved to {output_path}. Open this file in a web browser to view.")
