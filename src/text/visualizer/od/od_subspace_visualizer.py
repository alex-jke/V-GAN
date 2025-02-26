from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray


def plot_subspaces(subspaces: ndarray, proba: ndarray, output_dir: Path):
    """
    A function to plot the subspaces and their probabilities.
    Specifically, this method is designed to be used by the V_ODM class.
    It can be used to visualize the subspaces and their probabilities, that
    were used to perform the outlier detection. This is not a Visualizer
    class, as that class generated its own subspaces. This might not
    be representative of the actual subspaces used in the OD process.
    :param subspaces: The subspaces to plot as a two-dimensional ndarray.
    :param proba: The probabilities of the subspaces as a one-dimensional ndarray.
    :param output_dir: The directory to save the plots to as a Path object.
    """
    #for prob, subspace in zip(proba, subspaces):
    for i in range(proba.shape[0]):
        subspace = subspaces[i]
        prob = proba[i]
        plot, ax = plt.subplots()
        ax.plot(subspace)
        ax.set_title(f"Subspace with probability {prob}")
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        plt.savefig(output_dir / f"subspace_{i}_{prob}.png")
    #df = pd.DataFrame({"subspace": subspaces, "probability": proba})
    #df.to_csv(output_dir / "subspaces.csv")