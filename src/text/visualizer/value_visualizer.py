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
            subspaces = self.get_subspaces(samples, round=False)

            for i in range(samples):
                ax.plot(subspaces[i].cpu().detach().numpy(), label=f"Sample {i}")

        y_label_text = "Subspace value"

        # Plot the average subspace
        if samples == 0:
            ax.plot(self.avg_subspace.detach().cpu().numpy(), label="Average")
            y_label_text = "Average " + y_label_text

        ax.set_title(f"{y_label_text} per Dimension")
        ax.set_xlabel("Dimension")
        ax.set_ylabel(y_label_text)

        # Set the y-axis to be between 0 and 1
        ax.set_ylim(-0.05, 1.05)


        if samples < 11:
            ax.legend()
        postfix = "" if epoch == -1 else f"_{epoch}"
        output_path = self.output_dir / f"value_{samples}" / f"value{postfix}.png"
        if not os.path.exists(output_path.parent):
            os.makedirs(output_path.parent)
        plt.savefig(output_path)
