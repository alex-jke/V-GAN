import numpy as np
import pandas as pd

from text.visualizer.visualizer import Visualizer
from vmmd import model_eval


class SubspaceVisualizer(Visualizer):

    def visualize(self, samples: int = 1, epoch: int = 0):
        unique_subspaces = self.get_unique_subspaces()
        output_dir = self.output_dir / "subspaces"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        unique_subspaces.to_csv(output_dir / f"subspaces_{epoch}.csv", index=False)

    def get_unique_subspaces(self):
        u = self.get_subspaces(1000, round=True)
        unique_subspaces, proba = np.unique(
            np.array(u.to('cpu')), axis=0, return_counts=True)
        proba = proba / np.array(u.to('cpu')).shape[0]
        unique_subspaces = [str(unique_subspaces[i] * 1)
                            for i in range(unique_subspaces.shape[0])]

        subspace_df = pd.DataFrame({'subspace': unique_subspaces, 'probability': proba})
        # Sort by probability
        subspace_df = subspace_df.sort_values(by='probability', ascending=False)
        return subspace_df
