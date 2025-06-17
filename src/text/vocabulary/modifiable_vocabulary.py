from abc import abstractmethod, ABC

import pandas as pd

from text.vocabulary.vocabulary import Vocabulary


class ModifiableVocabulary(Vocabulary, ABC):
    @abstractmethod
    def add_text(self, words: str):
        pass

    @abstractmethod
    def add_word(self, word: str) -> bool:
        pass

    @abstractmethod
    def add_series(self, series: pd.Series):
        pass
