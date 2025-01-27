from text.Embedding.tokenizer import Tokenizer
from text.dataset.dataset import Dataset
import pandas as pd


class SyntheticDataset(Dataset):

    def __init__(self, number_of_samples: int, sample: str, tokenizer: Tokenizer, number_of_subspaces: int = 1):
        self.number_of_samples = number_of_samples
        self.number_of_subspaces = number_of_subspaces
        self.sample = sample
        self.tokenizer = tokenizer
        super().__init__()

    def get_possible_labels(self) -> list:
        return [0]

    def _import_data(self):

        amount_per_subspace = self.number_of_samples // self.number_of_subspaces
        tokenized_example = self.tokenizer.tokenize(self.sample)
        single_subspace_size = len(tokenized_example)
        zero_token = self.tokenizer.detokenize([0]) + " "
        sample_list = []
        for i in range(self.number_of_subspaces):
            for j in range(amount_per_subspace):
                sample = ""
                for _ in range(i):
                    for l in range(single_subspace_size):
                        sample += zero_token
                sample += self.sample
                sample_list.append(sample)


        df = pd.DataFrame(sample_list, columns=[self.x_label_name])
        df[self.y_label_name] = 0
        self.split(df)

    @property
    def name(self) -> str:
        return f"synthetic_{self.sample}_{self.number_of_subspaces}_{self.number_of_samples}"

    @property
    def x_label_name(self) -> str:
        return "x"

    @property
    def y_label_name(self):
        return "y"