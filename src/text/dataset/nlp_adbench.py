from abc import ABC
from typing import List

import pandas as pd

from datasets import load_dataset

from text.dataset.ag_news import AGNews
from text.dataset.dataset import Dataset, AggregatableDataset
from text.dataset.prompt import Prompt


class NLP_ADBench(AggregatableDataset, ABC):

    def __init__(self, dataset_name: str, prompt: Prompt):
        self.dataset_name = dataset_name
        self._assert_is_valid_name(dataset_name)
        super().__init__(prompt=prompt)

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

            # Shuffle the dataset as all outliers are concentrated, where the original labels were.
            test_dataset = test_dataset.sample(frac=1).reset_index(drop=True)
            train_dataset = train_dataset.sample(frac=1).reset_index(drop=True)

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
    def agnews(cls) :
        return NLPADBenchAGNews()

    @classmethod
    def n24news(cls):
        return NLPADBenchN24News()

    @classmethod
    def bbc(cls):
        return NLPADBenchBBC()

    @classmethod
    def email_spam(cls):
        return NLPADBenchEmailSpam()

    @classmethod
    def emotion(cls):
        return NLPADBenchEmotion()

    @classmethod
    def movie_review(cls):
        return NLPADBenchMovieReview()

    @classmethod
    def sms_spam(cls):
        return NLPADBenchSMSSpam()

    @classmethod
    def yelp_review_polarity(cls):
        return NLPADBenchYelpReviewPolarity()

    @staticmethod
    def possible_datasets():
        return ["agnews", "N24News", "bbc", "email_spam", "emotion", "movie_review", "sms_spam", "yelp_review_polarity"]

    @classmethod
    def get_all_datasets(cls) -> List[AggregatableDataset]:
        return [NLPADBenchAGNews(), NLPADBenchN24News(), NLPADBenchBBC(), NLPADBenchEmailSpam(), NLPADBenchEmotion(),
                NLPADBenchMovieReview(), NLPADBenchSMSSpam(), NLPADBenchYelpReviewPolarity()]

class NLPADBenchAGNews(NLP_ADBench):
    def __init__(self):
        super().__init__("agnews", AGNews.prompt)


class NLPADBenchN24News(NLP_ADBench):
    prompt = Prompt(
        sample_prefix="Description :",
        label_prefix="News type :",
        samples=["Mcdonald's to introduce a new burger in Germany.",
                 "Stocks rally as electronics get a tariff break."],
        labels=["Food", "Business"]
    )
    def __init__(self):
        super().__init__("N24News", self.prompt)

class NLPADBenchBBC(NLP_ADBench):
    prompt = Prompt(
        sample_prefix="text :",
        label_prefix="news type :",
        samples=["The stars from the movie 'The Breakfast Club' reunite for the fist time in 40 years.",
                 "Stocks rally as electronics get a tariff break."],
        labels=["Entertainment", "Business"]
    )
    def __init__(self):
        super().__init__("bbc", self.prompt)

class NLPADBenchEmailSpam(NLP_ADBench):
    prompt = Prompt(
        sample_prefix="email :",
        label_prefix="spam type :",
        samples=["Subject: YOU WON! Congratulations! You've won a $1,000 cash prize!",
                 "Subject: Submission Approved. Hi, your submission has been approved."],
        labels=["spam", "no spam"]
    )
    def __init__(self):
        super().__init__("email_spam", self.prompt)

class NLPADBenchEmotion(NLP_ADBench):
    prompt = Prompt(
        sample_prefix="text :",
        label_prefix="emotion :",
        samples=["I am so happy today, as I got a promotion.",
                        "I feel absolutely devastated after hearing the news."],
        labels=["happiness", "sadness"]
    )
    def __init__(self):
        super().__init__("emotion", self.prompt)

class NLPADBenchMovieReview(NLP_ADBench):
    prompt = Prompt(
        sample_prefix="review :",
        label_prefix="sentiment :",
        samples=["This movie was great!",
                 "I really did not like this movie."],
        labels=["positive", "negative"]
    )
    def __init__(self):
        super().__init__("movie_review", self.prompt)

class NLPADBenchSMSSpam(NLP_ADBench):
    prompt = Prompt(
        sample_prefix="sms:",
        label_prefix="spam type:",
        samples=["Congratulations! You've won a $1,000 cash prize! Click here to claim it now!",
                 "What r ur plans for tonight?"],
        labels=["spam", "no spam", ]
    )
    def __init__(self):
        super().__init__("sms_spam", self.prompt)

class NLPADBenchYelpReviewPolarity(NLP_ADBench):
    prompt = Prompt(
        sample_prefix="review :",
        label_prefix="sentiment :",
        samples=["This restaurant is amazing! The food was delicious and the service was excellent.",
                 "I had a terrible experience. The food was cold and the staff was rude."],
        labels=["positive", "negative"]
    )
    def __init__(self):
        super().__init__("yelp_review_polarity", self.prompt)


if __name__ == "__main__":
    dataset = NLP_ADBench.sms_spam()
    dataset._import_data()
    print(dataset.x_train)