from typing import List

import pandas as pd

from datasets import load_dataset
from text.dataset.dataset import Dataset


class NLP_ADBench(Dataset):

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self._assert_is_valid_name(dataset_name)
        super().__init__()

    def get_possible_labels(self) -> list:
        return [0, 1]

    def _import_data(self):
        dataset_path = self.resources_path / "NLP-ADBench"
        if not dataset_path.exists():
            dataset_path.mkdir(parents=True, exist_ok=True)
            dataset = load_dataset("kendx/NLP-ADBench")
            train_dataset = dataset["train"]
            train_dataset = pd.DataFrame(train_dataset)
            test_dataset = dataset["test"]
            test_dataset = pd.DataFrame(test_dataset)

            train_dataset.to_csv(dataset_path / "train.csv", index=False)
            test_dataset.to_csv(dataset_path / "test.csv", index=False)
        else:
            train_dataset = pd.read_csv(dataset_path / "train.csv")
            test_dataset = pd.read_csv(dataset_path / "test.csv")

        task_label = self._get_row()
        train_dataset_filtered = train_dataset[train_dataset["original_task"] == task_label].reset_index(drop=True)
        test_dataset_filtered = test_dataset[test_dataset["original_task"] == task_label].reset_index(drop=True)

        self.x_train = train_dataset_filtered[self.x_label_name]
        self.y_train = train_dataset_filtered[self.y_label_name]
        self.x_test = test_dataset_filtered[self.x_label_name]
        self.y_test = test_dataset_filtered[self.y_label_name]
        self.imported = True

    @property
    def name(self) -> str:
        return "NLP_ADBench " + self.dataset_name

    @property
    def x_label_name(self) -> str:
        return "text"

    @property
    def y_label_name(self):
        return "label"

    def _get_row(self):
        name_to_row_map = {
            "agnews": "AG News Classification",
            "N24News": "N24News: A New Dataset for Multimodal News Classification",
            "bbc": "BBC news category classification",
            "email_spam": "To check if an email is a spam",
            "emotion": "Emotion is a dataset of English Twitter messages with six basic emotions: anger, fear, joy, love, sadness, and surprise.",
            "movie_review": "For the task of binary sentiment classification: such like giving a review to decide if it is pos or neg.",
            "sms_spam": "SMS spam classification",
            "yelp_review_polarity": "Yelp reviews dataset consists of reviews from Yelp"
        }
        key = self.dataset_name
        return name_to_row_map[key]

    def _assert_is_valid_name(self, name: str):
       if name not in self.possible_datasets():
           raise ValueError(f"Invalid dataset. Possible dataset: {self.possible_datasets()}")

    @classmethod
    def agnews(cls):
        return cls("agnews")

    @classmethod
    def n24news(cls):
        return cls("N24News")

    @classmethod
    def bbc(cls):
        return cls("bbc")

    @classmethod
    def email_spam(cls):
        return cls("email_spam")

    @classmethod
    def emotion(cls):
        return cls("emotion")

    @classmethod
    def movie_review(cls):
        return cls("movie_review")

    @classmethod
    def sms_spam(cls):
        return cls("sms_spam")

    @classmethod
    def yelp_review_polarity(cls):
        return cls("yelp_review_polarity")

    @staticmethod
    def possible_datasets():
        return ["agnews", "N24News", "bbc", "email_spam", "emotion", "movie_review", "sms_spam", "yelp_review_polarity"]

    @classmethod
    def get_all_datasets(cls) -> List[Dataset]:
        return [cls(dataset) for dataset in cls.possible_datasets()]

if __name__ == "__main__":
    dataset = NLP_ADBench.agnews()
    dataset._import_data()
    print(dataset.x_train)