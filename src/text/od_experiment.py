import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd

from pyod.models.lunar import LUNAR as pyod_LUNAR
from pyod.models.ecod import ECOD as pyod_ECOD
from pyod.models.lof import LOF as pyod_LOF

from text.Embedding.bert import Bert
from text.Embedding.deepseek import DeepSeek1B
from text.Embedding.gpt2 import GPT2
from text.Embedding.huggingmodel import HuggingModel
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.wikipedia_slim import WikipediaPeopleDataset
from text.outlier_detection.VGAN_odm import VGAN_ODM
from text.outlier_detection.odm import OutlierDetectionModel
from text.outlier_detection.pyod_odm import LOF, LUNAR, ECOD, FeatureBagging
from text.outlier_detection.trivial_odm import TrivialODM
from text.visualizer.result_visualizer import ResultVisualizer


class Experiment:
    def __init__(self, dataset, emb_model, skip_error: bool = True, train_size: int = 2000, test_size: int = 200,
                 models: List[OutlierDetectionModel] = None, output_path: Path = None):
        """
        Initializes the experiment with a dataset, an embedding model, and error handling.
        """
        self.dataset = dataset
        self.emb_model = emb_model
        self.skip_error = skip_error

        # Experiment parameters
        self.partial_params: Dict = {
            "dataset": self.dataset,
            "model": self.emb_model,
            "train_size": train_size,  # TODO: add ability to not crash when chosen too large
            "test_size": test_size,
        }
        # Parameters used for models that require embedding.
        self.params: Dict = {**self.partial_params, "use_embedding": True}

        # Build the list of outlier detection models.
        self.models = models
        if models is None:
            self.models: List = self._build_models()

        # DataFrames to store experiment results and errors.
        self.result_df: pd.DataFrame = pd.DataFrame()
        self.error_df: pd.DataFrame = pd.DataFrame(columns=["model", "error"])

        # Determine the output directory.
        self.output_path = output_path
        if output_path is None:
            self.output_path: Path = self._get_output_path()

    def _build_models(self) -> List[OutlierDetectionModel]:
        """
        Builds and returns a list of outlier detection model instances.
        """
        bases = [pyod_LUNAR, pyod_ECOD, pyod_LOF]
        models = []

        # Base models that always use embedding.
        models.extend([
            LOF(**self.params),
            LUNAR(**self.params),
            ECOD(**self.params)
        ])

        # VGAN ODM models with both use_embedding False and True.
        models.extend([
            VGAN_ODM(**self.partial_params, base_detector=base, use_embedding=use_emb)
            for base in bases
            for use_emb in [False, True]
        ])

        # Trivial ODM models with different guess inlier rates.
        models.extend([
            TrivialODM(**self.partial_params, guess_inlier_rate=rate)
            for rate in [0.0, 0.5, 1.0]
        ])

        return models

    def _get_output_path(self) -> Path:
        """
        Constructs and returns the output path for saving results.
        """
        return Path(os.getcwd()) / 'results' / 'outlier_detection_noFB' / self.dataset.name / self.emb_model.model_name

    def _run_single_model(self, model) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Runs training, prediction, and evaluation for a single model.
        Returns a tuple (evaluation_results, error_record) where each is a DataFrame.
        """
        print(f"Testing model: {model.name}")
        try:
            model.start_timer()
            model.train()
            model.predict()
            model.stop_timer()
            evaluation = model.evaluate(self.output_path)
            return evaluation, None
        except Exception as e:
            if not self.skip_error:
                raise e
            error_record = pd.DataFrame({
                "model": [model.name],
                "error": [str(e)]
            })
            return pd.DataFrame(), error_record

    def _visualize_and_save_results(self) -> None:
        """
        Visualizes the results and saves both the result and error DataFrames to CSV files.
        """
        visualizer = ResultVisualizer(self.result_df, self.output_path)
        for column in self.result_df.columns:
            if column != "method":
                visualizer.visualize(x_column="method", y_column=column)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.result_df.to_csv(self.output_path / "results.csv", index=False)
        self.error_df.to_csv(self.output_path / "errors.csv", index=False)

    def run(self) -> None:
        """
        Executes the complete experiment pipeline.
        """
        # Import dataset data.
        self.dataset._import_data()  # Consider renaming to a public method if possible.

        # Run each model experiment.
        for model in self.models:
            evaluation, error = self._run_single_model(model)
            self.result_df = pd.concat([self.result_df, evaluation], ignore_index=True)
            if error is not None:
                self.error_df = pd.concat([self.error_df, error], ignore_index=True)

        print(self.result_df)
        self._visualize_and_save_results()


if __name__ == '__main__':
    datasets = [EmotionDataset(), IMBdDataset(), AGNews()]
    embedding_models = [GPT2(), Bert(), DeepSeek1B()]

    # Create and run an experiment for every combination of dataset and embedding model.
    for dataset in datasets:
        for emb_model in embedding_models:
            experiment = Experiment(dataset=dataset, emb_model=emb_model)
            experiment.run()