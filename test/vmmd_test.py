import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from models.Generator import GeneratorUpperSoftmax, GeneratorSigmoidAnnealing, GeneratorSpectralSigmoidSTE, \
    GeneratorSoftmaxAnnealing, GeneratorSoftmaxSTE, GeneratorSigmoidSTE, GeneratorSigmoidSoftmaxSTE, \
    GeneratorSigmoidSoftmaxSigmoid, GeneratorSigmoidSTEMBD, GeneratorSoftmaxSTESpectralNorm, GeneratorSoftmaxSTEMBD, \
    Generator, OldGeneratorUpperSoftmax
from vmmd import VMMD, model_eval


def generate_correlated_data(n_samples=1000):
    np.random.seed(42)
    # Group 1: dimensions 0-2 (highly correlated)
    latent1 = np.random.normal(0, 1, (n_samples, 1))
    group1 = latent1 + np.random.normal(0, .1, (n_samples, 3))
    # Group 2: dimensions 3-4 (highly correlated)
    latent2 = np.random.normal(0, 1, (n_samples, 1))
    group2 = latent2 + np.random.normal(0, .1, (n_samples, 2))
    # Noise: dimensions 5-9 (independent noise)
    noise = np.random.normal(0, 1, (n_samples, 5))
    data = np.hstack((group1, group2, noise))
    return data

def generate_subspace_data(n_samples=1000, F: float = 0.5):
    np.random.seed(42)
    samples_per_space = (int(n_samples * F), int(n_samples * (1-F)))
    # Group 1: points on the x-y plane
    group1 = np.random.normal(0, 1, (samples_per_space[0], 2))

    # Group 2: points on the z-axis
    group2 = np.random.normal(0,  1, (samples_per_space[1], 1))

    group1_z = np.zeros_like(np.arange(samples_per_space[0]).reshape(samples_per_space[0], 1))
    group2_xy = np.zeros_like(np.arange(samples_per_space[1] * 2).reshape(samples_per_space[1], 2))

    if F == 0.0:
        s2 = np.hstack((group2_xy, group2))
        data = s2
    elif F == 1.0:
        s1 = np.hstack((group1, group1_z))
        data = s1
    else:
        s1 = np.hstack((group1, group1_z))
        s2 = np.hstack((group2_xy, group2))
        data = np.vstack((s1, s2))
    return data

def save_results(df: DataFrame, name: str, base_path: Path) -> None:
    if not base_path.exists():
        base_path.mkdir(parents=True)
    df.to_csv(base_path / f"{name}.csv")

class VMMDTest(unittest.TestCase):

    def test_synthetic(self):
        samples = generate_correlated_data(2000)
        vmmd = VMMD(generator=GeneratorUpperSoftmax, weight=0.0, epochs=3000, lr=1e-2)
        vmmd.fit(samples)
        subspace_df = model_eval(vmmd, samples)

    def test_synthetic2(self):

        for generator in [OldGeneratorUpperSoftmax,
                          GeneratorSigmoidAnnealing,
                            GeneratorSoftmaxAnnealing, GeneratorUpperSoftmax, GeneratorSoftmaxSTE, GeneratorSigmoidSTE, GeneratorSpectralSigmoidSTE, GeneratorSigmoidSoftmaxSTE, GeneratorSigmoidSoftmaxSigmoid, #GeneratorSigmoidSTEMBD,
                            GeneratorSoftmaxSTESpectralNorm,
                          #GeneratorSoftmaxSTEMBD
                          ]:
            subspace_probas = DataFrame({"S_1": [], "S_2": [], "F_1": [], "F_2": []})
            all_subspaces = DataFrame()
            base_path = Path(os.path.dirname(__file__)) / "results" / "F-Test_4" / f"{generator.__name__} no_penalty"
            if base_path.exists():
                print(f"Skipping generator {generator.__name__}")
                continue
            try:
                for F in [.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:

                    samples = generate_subspace_data(10000, F)
                    vmmd = VMMD(generator=generator, weight=0.0, epochs=3000, lr=1e-2)
                    vmmd.fit(samples)
                    subspace_df = model_eval(vmmd, samples)
                    subspace_df["F"] = F
                    s1 = 0.0
                    s2 = 0.0
                    all_subspaces = pd.concat([all_subspaces, subspace_df], ignore_index=True)
                    for row in subspace_df.itertuples():
                        subspace = row[1]
                        proba = row[2]
                        if str(subspace) == "[1. 1. 0.]":
                            s1 = proba
                        elif str(subspace) == "[0. 0. 1.]":
                            s2 = proba
                    current_probas = DataFrame({"S_1": [s1], "S_2": [s2], "F_1": [F], "F_2": [1 - F]})
                    subspace_probas = pd.concat([subspace_probas, current_probas], ignore_index=True)
                    print(all_subspaces)
                    print(subspace_probas)
            except:
                continue

            save_results(all_subspaces, "all_subspaces", base_path)
            save_results(subspace_probas, "subspace_probas", base_path)
