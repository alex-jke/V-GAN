from abc import ABC

import numpy as np
import pandas as pd

from text.visualizer.visualizer import Visualizer
from vmmd import model_eval


class SubspaceVisualizer(Visualizer, ABC):
    def __init__(self, **kwargs):
        self.subspace_col = 'subspace'
        self.probability_col = 'probability'
        super().__init__(**kwargs)

    def get_unique_subspaces(self):
        u = self.get_subspaces(1000, round=True)
        unique_subspaces, proba = np.unique(
            np.array(u.to('cpu')), axis=0, return_counts=True)
        proba = proba / np.array(u.to('cpu')).shape[0]
        unique_subspaces = [str(unique_subspaces[i] * 1)
                            for i in range(unique_subspaces.shape[0])]

        subspace_df = pd.DataFrame({self.subspace_col: unique_subspaces, self.probability_col: proba})
        # Sort by probability
        subspace_df = subspace_df.sort_values(by=self.probability_col, ascending=False)
        return subspace_df
