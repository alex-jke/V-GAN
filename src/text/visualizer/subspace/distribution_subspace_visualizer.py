from matplotlib import pyplot as plt

from text.visualizer.subspace.subspace_visualizer import SubspaceVisualizer

class DistributionSubspaceVisualizer(SubspaceVisualizer):

    def visualize(self, samples: int = 1, epoch: int = 0):
        """
        Visualizes the distribution of the subspaces.
        """
        subspaces = self.get_unique_subspaces()
        output_dir = self.output_dir / "subspaces"
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        # Plot the probability distribution of the subspaces
        plot, ax = plt.subplots()
        ax.bar(subspaces[self.subspace_col], subspaces[self.probability_col], color=self.vgan_color)
        ax.set_xticklabels([])