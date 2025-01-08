import os

from text.visualizer.visualizer import Visualizer
import matplotlib.pyplot as plt


class ValueVisualizer(Visualizer):

    def visualize(self, samples: int = 1, epoch: int = -1):
        """
        Visualizes the data with the values.
        :param samples: The number of samples to visualize.
        """
        plot, ax = plt.subplots()

        if samples > 0:
            # Plot sample amount of subspaces
            subspaces = self.get_subspaces(samples)

            for i in range(samples):
                ax.plot(subspaces[i].cpu().numpy(), label=f"Sample {i}")

        # Plot the average subspace
        ax.plot(self.avg_subspace.cpu().numpy(), label="Average")

        ax.set_title("Average Subspace Value per Token position")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Average Subspace Value")

        # Set the y-axis to be between 0 and 1
        ax.set_ylim(-0.05, 1.05)

        ax.legend()
        postfix = "" if epoch == -1 else f"_{epoch}"
        output_path = self.output_dir / "value" / f"value{postfix}.png"
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent)
        plt.savefig(output_path)
