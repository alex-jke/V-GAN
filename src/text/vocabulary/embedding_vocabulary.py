import numpy as np
import pandas as pd

from Embedding.embedding import Embedding
from vocabulary.modifiable_vocabulary import ModifiableVocabulary


class EmbeddingVocabulary(ModifiableVocabulary):
    def __init__(self, embedding: Embedding):
        self.embedding = embedding
        self.word2vec = {}
        self.vec2word = {}
        self.special_characters = ['\n', '\r', '\t', '.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '<', '>',
                              '|', '\'\'',
                              '\\', '/', '_', '+', '=', '*', '&', '^', '%', '$', '#', '@', '~', '`', '"', '-']

    def add_word(self, word: str) -> bool:
        if word not in self.word2vec:
            embedding = self.embedding.embed(word)
            self.word2vec[word] = embedding
            self.vec2word[embedding.tostring()] = word
            return True

        return False

    def add_words(self, words: list[str]):
        embeddings = self.embedding.embed_words(words)
        for i, word in enumerate(words):
            embedding_vec = embeddings[i]
            self.word2vec[word] = embedding_vec
            self.vec2word[embedding_vec.tostring()] = word #todo: replace tostring with tobytes

    def get_word(self, vec: np.ndarray) -> str | None:
        if vec.tostring() not in self.vec2word:
            return None
        return self.vec2word[vec.tostring()]

    def get_vec(self, word: str) -> np.ndarray:
        if word not in self.word2vec:
            self.add_word(word)
        return self.word2vec[word]

    def containsWord(self, word: str) -> bool:
        return word in self.word2vec

    def containsVec(self, vec: np.ndarray) -> bool:
        return vec.tostring() in self.vec2word

    def get_size(self) -> int:
        return len(self.word2vec)

    def get_words(self) -> list[str]:
        return list(self.word2vec.keys())

    def get_vecs(self) -> list[np.ndarray]:
        return list(self.word2vec.values())

    def add_text(self, words: str, should_split=True):
        if should_split:
            for char in self.special_characters:
                words = words.replace(char, '')
            words = words.split(" ")
            self.add_words(words)

            return
        self.add_word(words)

    def add_series(self, series: pd.Series):

        #create a list of all the words in the series
        """series = series.apply(self.format_text)
        words = series.str.cat(sep=' ')
        self.add_words(words)"""
        for text in series:
            self.add_text(text)

    def format_text(self, text: str) -> str:
        for char in self.special_characters:
            text = text.replace(char, '')
        return text
