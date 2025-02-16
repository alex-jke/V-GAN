import torch
from torch import Tensor

from text.Embedding.huggingmodel import HuggingModel
from text.dataset.dataset import Dataset
from text.dataset_converter.dataset_embedder import DatasetEmbedder
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer


class DatasetProcessor:
    """
    Handles tokenization and optional embedding of the dataset.
    """
    def __init__(self, dataset: Dataset, model: HuggingModel, sequence_length: int | None, samples: int, pre_embed: bool,
                 inlier_labels: list):
        self.dataset = dataset
        self.model = model
        # Ensure sequence_length does not exceed model maximum
        self.sequence_length: int | None = sequence_length
        self.samples = samples
        self.device = get_device()
        self.pre_embed = pre_embed
        self.inlier_labels = inlier_labels
        if inlier_labels is None:
            self.inlier_labels = dataset.get_possible_labels()[:1]

    def process(self) -> (Tensor, Tensor):
        # Tokenize the data
        tokenizer = DatasetTokenizer(tokenizer=self.model, dataset=self.dataset, max_samples=self.samples)
        #labels = self.dataset.get_possible_labels()[:1]


        if not self.pre_embed:
            data, _ = tokenizer.get_tokenized_training_data(self.inlier_labels)
            # Take only the first sequence_length tokens per sample

            if self.sequence_length is not None:
                first_part = data[:, :self.sequence_length].to(self.device)
            else:
                first_part = data.to(self.device)
            if self.device.type == 'cuda':
                first_part = first_part.float()
            normalized = first_part  # tokens are not normalized
        else:
            # If using embeddings, embed and then normalize
            embedder = DatasetEmbedder(dataset=self.dataset, model=self.model)
            first_part, _ = embedder.embed(train=True, samples=self.samples, labels=self.inlier_labels)
            normalized = torch.nn.functional.normalize(first_part, p=2, dim=1)

        return first_part, normalized

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        return torch.device('mps:0')
    else:
        return torch.device('cpu')