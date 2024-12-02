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

    def __init__(self, model, tokenized_data: Tensor, tokenizer: Tokenizer): #todo: replace subspaces with model.
        super().__init__()
        self.num_subspaces = 500
        # Tensor is of the shape (num_subspaces, sequence_length)
        subspaces: Tensor = model.generate_subspaces(self.num_subspaces)
        #unique_subspaces, proba = np.unique(np.array(subspaces.to('cpu')), axis=0, return_counts=True)
        #proba = proba / np.array(subspaces.to('cpu')).shape[0]

        #unique_subspaces = [Tensor([int(boolean) * proba[i] for boolean in boolean_list]) for i, boolean_list in enumerate(unique_subspaces.tolist())]

        #self.subspace_df: pd.DataFrame = pd.DataFrame({SUBSPACE_COLUMN: unique_subspaces, SUBSPACE_PROBABILITY_COLUMN: proba})
        self.avg_subspace: Tensor = subspaces.sum(dim=1) / self.num_subspaces

        self.tokenized_data = tokenized_data
        self.tokenizer: Tokenizer = tokenizer

    def visualize(self, samples: int = 1):
        """
        Visualizes the data with alpha characters.
        :param samples: The number of samples to visualize.
        """
        sample_data = self.tokenized_data[:samples] #todo: get random samples
        most_selected = torch.argsort(self.avg_subspace, descending=True)[:10]
        print("indices and value of most selected features:", [(int(i), float(self.avg_subspace[i])) for i in most_selected])
        print("average subspace:", self.avg_subspace)
        #fig, ax = plt.subplots()
        x_dim = 15
        y_dim = 20
        fig = plt.figure(figsize=(x_dim, y_dim))
        y_pos = 1
        for i in range(samples):
            print("Sample", i)
            int_list = [int(number) for number in sample_data[i].tolist()]
            print("Original:", self.tokenizer.detokenize(int_list))
            strings = [self.tokenizer.detokenize(token) for token in int_list]

            # Normalize the average subspace values
            values = (self.avg_subspace - self.avg_subspace.min()) / (self.avg_subspace.max() - self.avg_subspace.min())
            # Create a figure and axis

            # Loop through strings and their transparency values
            x_pos = 0.01  # Starting x position
            y_pos -= y_dim / 1000  # Adjusted vertical spacing
            for j, (string, alpha) in enumerate(zip(strings, values)):
                if string == '[PAD]':
                    continue
                if x_pos > 0.95:
                    y_pos -= y_dim / 2000 # Adjusted vertical spacing
                    x_pos = 0.01
                fig.text(
                    x_pos,
                    y_pos,  # Adjusted vertical spacing
                    string,
                    fontsize=12,
                    alpha=float(alpha),
                    ha='left',
                    va='center',
                    family='monospace'  # Fixed-width font for better spacing
                )
                x_pos += x_dim / 2000 * len(string) + x_dim / 2000

            # Hide axes
            plt.axis('off')

        # Show the plot
        plt.tight_layout()
        path = Path(os.getcwd()) / "results" / "alpha_visualization.png"
        plt.savefig(path)
        plt.show()

