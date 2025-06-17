import enchant
import numpy as np

from text.vocabulary.vocabulary import Vocabulary


class EnglishVocabulary(Vocabulary):
    def __init__(self):
        self.dict = enchant.Dict("en_US")

    def get_word(self, vec: np.ndarray) -> str | None:
        pass

    def get_vec(self, word: str) -> np.ndarray:
        pass

    def containsWord(self, word: str) -> bool:
        pass

    def containsVec(self, vec: np.ndarray) -> bool:
        pass

    def get_size(self) -> int:
        pass

    def get_words(self) -> list[str]:
        pass
    def get_vecs(self) -> list[np.ndarray]:
        pass



if __name__ == '__main__':
    vocab = EnglishVocabulary()
    print(vocab.get_words())
    #print(vocab.get_size())
    #print(vocab.containsWord("hello"))
    #print(vocab.containsWord("helo"))
    #print(vocab.containsWord("hola"))