import os
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
import torch
import random

from pyod.models.lunar import LUNAR as pyod_LUNAR
from pyod.models.ecod import ECOD as pyod_ECOD
from pyod.models.lof import LOF as pyod_LOF

from text.Embedding.bert import Bert
from text.Embedding.deepseek import DeepSeek1B, DeepSeek14B, DeepSeek7B
from text.Embedding.gpt2 import GPT2
from text.Embedding.huggingmodel import HuggingModel
from text.UI import cli
from text.UI.cli import ConsoleUserInterface
from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.nlp_adbench import NLP_ADBench
from text.dataset.wikipedia_slim import WikipediaPeopleDataset
from text.outlier_detection import VGAN_odm
from text.outlier_detection.VGAN_odm import VGAN_ODM, DistanceVGAN_ODM, EnsembleVGAN_ODM
from text.outlier_detection.odm import OutlierDetectionModel
from text.outlier_detection.pyod_odm import LOF, LUNAR, ECOD, FeatureBagging
from text.outlier_detection.space.embedding_space import EmbeddingSpace
from text.outlier_detection.space.token_space import TokenSpace
from text.outlier_detection.trivial_odm import TrivialODM
from text.result_aggregator import ResultAggregator
from text.visualizer.result_visualizer import ResultVisualizer


class Experiment:
    def __init__(self, dataset, emb_model, skip_error: bool = True, train_size: int = -1, test_size: int = -1,
                 models: List[OutlierDetectionModel] = None, output_path: Path = None, experiment_name: str = None,
                 use_cached: bool = False, run_cachable: bool = False):
        """
        Initializes the experiment.
        : param dataset: The dataset to use for the experiment.
        : param emb_model: The embedding model to use for the experiment.
        : param skip_error: Whether to skip errors and continue running the experiment.
        : param train_size: The number of training samples to use for the experiment -1 means all samples will be used.
        : param test_size: The number of testing samples to use for the experiment -1 means all samples will be used.
        : param models: A list of OutlierDetectionModel instances to use for the experiment. If None, a default list will be used.
        : param output_path: The path to save the results of the experiment. If None, a default path will be used.
        : param experiment_name: The name of the experiment. If None, a default name will be used.
        : param use_cached: Whether to use cached data for the experiment. Using caching might distort the time measurements.
        : param run_cachable: Whether to run the cachable part of the experiment. If False, the experiment will run all models. This allows for quicker testing.
        """
        """
        Initializes the experiment
        """
        self.dataset = dataset
        self.emb_model = emb_model
        self.skip_error = skip_error
        self.experiment_name = "experiment_od" if experiment_name is None else experiment_name
        self.use_cached = use_cached
        self.run_cachable = run_cachable

        # Experiment parameters
        self.token_params: Dict = {
            "dataset": self.dataset,
            "use_cached": use_cached,
            "space": TokenSpace(model=emb_model, train_size=train_size, test_size=test_size)
        }
        self.emb_params: Dict = {
            "dataset": self.dataset,
            "use_cached": use_cached,
            "space": EmbeddingSpace(model=emb_model, train_size=train_size, test_size=test_size)
        }

        # Determine the output directory.
        self.output_path = output_path
        if output_path is None:
            self.output_path: Path = self._get_output_path()

        # Build the list of outlier detection models.
        self.models = models
        if models is None:
            self.models: List = self._build_models()

        # DataFrames to store experiment results and errors.
        self.result_df: pd.DataFrame = pd.DataFrame()
        self.comon_metrics: pd.DataFrame = pd.DataFrame()
        self.error_df: pd.DataFrame = pd.DataFrame(columns=["model", "error"])

        self.ui = cli.get()

    @classmethod
    @property
    def result_csv_name(cls) -> str:
        return "results.csv"

    @classmethod
    @property
    def comon_metrics_name(cls) -> str:
        return "comon_metrics.csv"

    def _build_models(self) -> List[OutlierDetectionModel]:
        """
        Builds and returns a list of outlier detection model instances.
        """
        bases = [
            pyod_LUNAR,
            #pyod_ECOD,
            pyod_LOF]
        models = []

        # Base models that always use embedding.
        models.extend([
            LOF(**self.emb_params),
            LUNAR(**self.emb_params),
            ECOD(**self.emb_params)
        ])

        # VGAN ODM models with both use_embedding False and True.
        params = [self.emb_params] if self.run_cachable else [self.token_params, self.emb_params]
        model_types = [VGAN_odm.VMMD]#[VGAN_odm.VGAN, VGAN_odm.VMMD]

        # VGAN ODM with both ensemble outlier detection, and subspace distance, only using pre-embedded, as euclidian
        # distance does not make sense for tokens.
        models.extend(
            [VGAN_ODM(**self.emb_params, base_detector=base,
                              output_path = self.output_path, model_type=model_type)
            for base in bases
            for model_type in model_types]
        )

        # VGAN ODM on the token space only using the ensemble method.
        if not self.run_cachable:
            models.extend([EnsembleVGAN_ODM(**self.token_params, base_detector=base,
                                    output_path=self.output_path) for base in bases])


        # VGAN ODM with only ensemble outlier detection.
        models.extend([EnsembleVGAN_ODM(**param, output_path=self.output_path, model_type=model_type)
                       for param in params
                       for model_type in model_types
                       ])

        # VGAN ODM with only subspace distance.
        models.extend([DistanceVGAN_ODM(**self.emb_params, output_path=self.output_path,
                                        model_type=model_type)
                       for model_type in model_types
                       ])

        models.extend([
            FeatureBagging(**param, base_detector=base)
            for base in bases
            for param in params
        ])



        # Trivial ODM models with different guess inlier rates.
        models.extend([
            TrivialODM(**self.token_params, guess_inlier_rate=rate)
            for rate in [0.0, 0.5, 1.0]
        ])

        return models

    def _get_output_path(self) -> Path:
        """
        Constructs and returns the output path for saving results.
        """
        return Path(os.getcwd()) / 'results' / "outlier_detection" /self.experiment_name / self.dataset.name / self.emb_model.model_name

    def _run_single_model(self, model) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Runs training, prediction, and evaluation for a single model.
        Returns a tuple (evaluation_results, error_record) where each is a DataFrame.
        """
        try:
            #model.start_timer()
            model.train()
            model.predict()
            #model.stop_timer()
            evaluation, self.comon_metrics = model.evaluate(self.output_path)
            print(f" | finished successfully (auc: {float(evaluation['auc']):>1.3f}).")
            return evaluation, None
        except Exception as e:
            if not self.skip_error:
                raise e
            error_record = pd.DataFrame({
                "model": [model.name],
                "error": [str(e)],
                "traceback": [str(traceback.format_exc())]
            })
            print(f"{model.name} encountered an error.")
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
        self.result_df.to_csv(self.output_path / self.result_csv_name, index=False)
        self.error_df.to_csv(self.output_path / "errors.csv", index=False)
        if not (self.output_path / self.comon_metrics_name).exists():
            self.comon_metrics.to_csv(self.output_path / self.comon_metrics_name, index=False)

    def _filter_for_not_run(self):
        result_path = self.output_path / self.result_csv_name
        if not result_path.exists():
            return
        result_df = pd.read_csv(result_path)
        not_run = []
        for model in self.models:
            run = result_df[model.method_column].tolist()
            if model._get_name() not in run:
                not_run.append(model)
        #print(f"Not run: {len(not_run)} models")
        random.shuffle(not_run)
        self.models = not_run
        self.result_df = result_df


    def run(self) -> None:
        """
        Executes the complete experiment pipeline.
        """
        # Import dataset data.
        self.dataset._import_data()  # Consider renaming to a public method if possible.

        self._filter_for_not_run()

        # Run each model experiment.
        with self.ui.display():
            for model in self.models:
                self.ui.update(f"Running model {model.__class__.__name__}")
                evaluation, error = self._run_single_model(model)
                self.result_df = pd.concat([self.result_df, evaluation], ignore_index=True)
                if error is not None:
                    self.error_df = pd.concat([self.error_df, error], ignore_index=True)
                    if not self.output_path.exists():
                        self.output_path.mkdir(parents=True, exist_ok=True)
                    self.error_df.to_csv(self.output_path / "errors.csv", index=False)
                    continue
                self._visualize_and_save_results()
                del model


        self._visualize_and_save_results()

def aggregate_results():
    aggregator = ResultAggregator(
        common_metrics_name=Experiment.comon_metrics_name,
        result_name=Experiment.result_csv_name
    )
    aggregator.run_aggregation()

if __name__ == '__main__':
    datasets = [
                   AGNews(),
                   IMBdDataset(),
                   EmotionDataset(),
                   ] + NLP_ADBench.get_all_datasets()
    embedding_models = [DeepSeek1B,
                        #GPT2, Bert,
                        #DeepSeek7B
                        ]

    # Python garbage collection does not collect garbage, that is clear the created models and datasets.
    # Thus, eventually, the program will crash.
    # The Experiment class checks, which models and datasets have been run. This approach guarantees, that
    # every model and dataset can run at some point.
    random.shuffle(embedding_models)
    #random.shuffle(datasets)

    ui = cli.get()
    train_size = 10_000
    test_size = 10_000

    # Create and run an experiment for every combination of dataset and embedding model.
    with ui.display():
        for dataset in datasets:
            ui.update(f"dataset {dataset.name}")
            with ui.display():
                for emb_model_cls in embedding_models:
                    emb_model = emb_model_cls()
                    ui.update(f"embedding model {emb_model.model_name}")
                    experiment = Experiment(dataset=dataset, emb_model=emb_model, train_size=train_size, test_size=test_size,
                                            experiment_name=f"0.272_small", use_cached=True,
                                            run_cachable=False)
                    experiment.run()
                    aggregate_results()
                    del emb_model
                    del experiment

                    if torch.cuda.is_available():
                        memory_used_before = torch.cuda.memory_allocated()
                        torch.cuda.empty_cache()
                        memory_used_after = torch.cuda.memory_allocated()
                        print(f"freed cuda cache: {memory_used_before} -> {memory_used_after}")