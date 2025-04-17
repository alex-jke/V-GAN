import os
import unittest
from pathlib import Path
from typing import Type, List

import pandas as pd
import torch

from text.Embedding.huggingmodel import HuggingModel
from text.Embedding.LLM.llama import LLama3B
from text.Embedding.unification_strategy import UnificationStrategy
from text.dataset.ag_news import AGNews
from text.dataset.dataset import AggregatableDataset
from text.dataset.emotions import EmotionDataset
from text.dataset.nlp_adbench import NLP_ADBench
from text.dataset.prompt import Prompt
from text.outlier_detection.pyod_odm import LUNAR, BasePyODM, LOF, ECOD
from text.outlier_detection.space.embedding_space import EmbeddingSpace
from text.outlier_detection.space.word_space import WordSpace

DATASET = "dataset"
EMB_MODEL = "emb_model"
PROMPT = "prompt"

class EmbTest(unittest.TestCase):

    def run_comparison(self, dataset: AggregatableDataset, base: Type[BasePyODM] = LUNAR, model: HuggingModel = None):
        if model is None:
            model = LLama3B()
        train_size = 15_000
        test_size = 3_000
        word_space = WordSpace(strategy=UnificationStrategy.TRANSFORMER, model=model, test_size=test_size,
                               train_size=train_size)
        embedding_space = EmbeddingSpace(model=model, train_size=train_size, test_size=test_size)

        shared_params = {
            "dataset": dataset,
            "use_cached": True,
        }

        word_lunar = base(
            space=word_space,
            **shared_params,
        )
        emb_lunar = base(
            space=embedding_space,
            **shared_params
        )
        word_lunar.train()
        word_lunar.predict()
        word_metrics = word_lunar.evaluate()[0]
        word_metrics["type"] = "t_agg"

        emb_lunar.train()
        emb_lunar.predict()
        emb_metrics = emb_lunar.evaluate()[0]
        emb_metrics["type"] = "avg"

        comparison_df = pd.concat([emb_metrics, word_metrics], ignore_index=True)
        comparison_df[DATASET] = dataset.name
        comparison_df["train_test_size"] = f"{train_size}_{test_size}"
        comparison_df[EMB_MODEL] = model.model_name
        comparison_df[PROMPT] = dataset.prompt.full_prompt

        print(comparison_df)

        return comparison_df

        #print("Word LUNAR metrics: ", word_metrics.to_markdown())
        #print("Embedding LUNAR metrics: ", emb_metrics.to_markdown())


    def test_emotions(self):
        dataset = EmotionDataset()
        self.run_comparison(dataset)

    def test_ag_news(self):
        dataset = AGNews()
        self.run_comparison(dataset)

    def run_all(self, output_path: Path, model: HuggingModel):
        datasets = NLP_ADBench.get_all_datasets()
        datasets.sort(key=lambda d: d.average_length)
        for dataset in datasets:
            for base in [LUNAR, LOF, ECOD]:

                if output_path.exists():
                    results_df = pd.read_csv(output_path)
                    already_run = results_df[DATASET].tolist()
                    bases = results_df["base"].tolist()
                    emb_models = results_df[EMB_MODEL].tolist()
                    if (dataset.name, base.__name__, model.model_name) in zip(already_run, bases, emb_models):
                        print(f"Skipping dataset {dataset.name} with base {base.__name__}, already present in results.")
                        continue
                try:
                    result = self.run_comparison(dataset, base, model)
                    result.to_csv(output_path, mode="a", header=not output_path.exists())
                except Exception as e:
                    print(f"An error occurred running {dataset.name} + {base.__name__}. Skipping. (error: {e}")
                    continue

    def run_different_prompts(self, dataset: AggregatableDataset, prompts: List[Prompt], output_path: Path, model: HuggingModel):
        default_prompt = dataset.prompt
        prompts = [default_prompt] + prompts
        for prompt in prompts:
            dataset.prompt = prompt
            result = self.run_comparison(dataset, model=model)
            result.to_csv(output_path, mode="a", header=not output_path.exists())

    def compare_sms_spam_prompts(self, output_path: Path, model: HuggingModel):
        dataset = NLP_ADBench.sms_spam()
        prompts = [
            Prompt(
                sample_prefix="sms :",
                label_prefix="spam type :",
                samples=["Congratulations! You've won a $1,000 cash prize!",
                         "What is our plan for tonight?.",
                         "Wht r ur plans 4 2night"],
                labels=["spam", "no spam", "no spam"] #TODO: this yields a little higher auc
            )
        ]
        self.run_different_prompts(dataset, prompts, output_path, model)


    def test_nlpadbench(self):

        path = Path(os.path.dirname(__file__)).parent.parent / "results" / "embedding_test"
        path.mkdir(parents=True, exist_ok=True)
        model = LLama3B()
        output_path = path / "results.csv"
        with torch.no_grad():
            self.run_all(output_path, model)
        #self.compare_sms_spam_prompts(output_path, model)





