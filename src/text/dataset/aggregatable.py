from abc import ABC, abstractmethod
from typing import List


class Aggregatable(ABC):
    """
    This is an interface that defines the interface for all aggregatable datasets.
    An aggregatable dataset is a dataset that can be aggregated to a single vector by an LLM.
    This is useful as it can provide a more meaningful aggregation of the samples, than
    averaging over the embeddings.
    A prefix is used to provide context to the LLM, so that it can better understand the samples.
    In other words, the prefix allows in-context learning.
    The postfix is then used to guid the LLM to the correct label.
    An example of such would be the Emotion dataset, where the samples are tweets and the labels are the emotions.
    Here the prefix could consist of two samples, where one might be sad and the other happy, with the corresponding labels.
    The suffix could then be "label: " or "emotion: ".
    """

    @abstractmethod
    def prefix(self) -> List[str]:
        """
        Returns the prefix of the dataset. Can include multiple samples.
        Important, is that the prefix is not too long, as it will be used in the context of the LLM.
        Too long prefixes will lead to a decrease in performance, as the memory of the GPU is rather limited.
        Too short prefixes will lead to a decrease in performance, as the LLM will not be able to learn from the samples.
        :return: The prefix, that will be prepended to all samples during embedding.
            It should be a list of strings, where each string is a word.
        """
        pass

    @abstractmethod
    def suffix(self) -> List[str]:
        """
        Returns the suffix of the dataset. It should be short and precise.
        It should guide the LLM to the correct label.
        :return: The suffix, that will be appended to all samples during embedding.
            It should be a list of strings, where each string is a word.
        """
        pass