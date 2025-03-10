import os
from pathlib import Path
from typing import List

import torch
from matplotlib import pyplot as plt
from torch import Tensor

from colors import VGAN_GREEN_RGB
from .visualizer import Visualizer
from ..UI.cli import ConsoleUserInterface

SUBSPACE_COLUMN = 'subspace'
SUBSPACE_PROBABILITY_COLUMN = 'probability'


class AlphaVisualizer(Visualizer):
    """
    Class for visualizing data with using the alpha in text.
    """

    def __init__(self, model, tokenized_data, tokenizer, path):
        self.samples = []
        super().__init__(model, tokenized_data, tokenizer, path)

    def _get_strings(self, token_list: list):
        if isinstance(token_list[0], str):
            return token_list
        strings = []
        for token in token_list:
            if token != self.tokenizer.padding_token:
                strings.append(self.tokenizer.detokenize([token]))
            else:
                strings.append("_")
        return strings

    def _convert_to_strings(self, tokens: Tensor) -> List[List[str]]:
        samples = []
        for i in range(tokens.size(0)):
            token_list = [int(token) for token in tokens[i].tolist()]

            if i >= len(self.samples):
                strings = self._get_strings(token_list)
                self.samples.append(strings)
                samples.append(strings)
        return samples

    def export_html(self, sample_data: Tensor | List[List[str]], subspaces: Tensor, folder_appendix: str, epoch: int = -1, normalize:bool = True):
        if self.tokenizer is not None:
            padding_token = self.tokenizer.detokenize([self.tokenizer.padding_token])
        ui = ConsoleUserInterface()
        if isinstance(sample_data, Tensor):
            samples: List[List[str]] = self._convert_to_strings(sample_data)
        else:
            samples: List[List[str]] = sample_data
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

        ui.update(f"Visualizing samples: ")
        #for i in range(sample_data.size(0)):
        for i in range(len(samples)):
            sample = samples[i]

            sample_length = len(sample)

            # Normalize the average subspace values
            values = subspaces[i][:sample_length]
            max, min = values.max(), values.min()
            if max != min and normalize:
                values = (values - min) / (max - min)
                # print("Normalized values:", values)

            html_content += f"<div><strong>Sample {i + 1}:</strong></div>"

            # Loop through strings and their transparency values
            for string, alpha in zip(sample, values):
                red = int(VGAN_GREEN_RGB[0] * alpha)
                green = int(VGAN_GREEN_RGB[1] * alpha)
                blue = int(VGAN_GREEN_RGB[2] * alpha)
                html_content += (f'<span class="token" style="color: rgba('
                                 f'{red}, {green}, {blue}, 1);">{string}</span>')
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
