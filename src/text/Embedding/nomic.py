from typing import List

import numpy as np
import ollama

from Embedding.embedding import Embedding

from vocabulary.vocabulary import Vocabulary

from sklearn.metrics.pairwise import cosine_similarity


class NomicEmbedding(Embedding):

    model = 'nomic-embed-text'

    def __init__(self, vocab: Vocabulary = None):
        self.vocab = vocab

    def set_vocabulary(self, vocab: Vocabulary):
        self.vocab = vocab

    def embed(self, data) -> np.ndarray:
        #embeddings = ollama.embeddings(model='nomic-embed-text', prompt=data)
        embedding = ollama.embed(model='nomic-embed-text', input=data)
        #return np.array(embeddings)
        return np.array(embedding['embeddings'])

    def embed_words(self, words: List[str]) -> List[np.ndarray]:
        embeddingsDict = ollama.embed(model=self.model, input=words)
        embeddings = embeddingsDict['embeddings']
        #print(embeddingsDict['total_duration'] / 1_000_000_000, "s\t", "embedded: ", words, "in")
        embeddings = [np.array(embedding) for embedding in embeddings]
        return embeddings

    def decode(self, target_embedding: np.ndarray) -> str:
        vocab_embeddings = self.vocab.get_vecs()  # Retrieve the vocabulary embeddings

        # Ensure target_embedding is 2D
        target_embedding = target_embedding.reshape(1, -1) if target_embedding.ndim == 1 else target_embedding

        closest_index = -1
        max_similarity = -1  # Initialize with a low similarity score

        for i, vec in enumerate(vocab_embeddings):
            # Check if vec is empty or has mismatched dimensions
            if vec is None or vec.size == 0 or vec.ndim == 0 or vec.shape[0] == 0:
                continue  # Skip any empty or invalid embeddings

            # Ensure vec is reshaped to 2D if needed
            vec = vec.reshape(1, -1) if vec.ndim == 1 else vec

            # Check if dimensions of vec and target_embedding match
            if vec.shape[1] != target_embedding.shape[1]:
                continue  # Skip vectors that don't match target dimensionality

            # Compute cosine similarity
            similarity = cosine_similarity(target_embedding, vec)[0, 0]

            # Update closest index if a higher similarity is found
            if similarity > max_similarity:
                max_similarity = similarity
                closest_index = i

        if closest_index == -1:
            raise ValueError("No valid embeddings found in vocabulary for comparison.")

        # Retrieve and return the word associated with the closest embedding
        return self.vocab.get_word(vocab_embeddings[closest_index])

if __name__ == '__main__':
    nomic = NomicEmbedding()
    #print(nomic.embed("Bachelor Thesis"))
    #print(nomic.embed_words(["Bachelor", "Thesis"]))
    #print(nomic.decode(nomic.embed("Bachelor Thesis")))
    nomic.model = 'llama3.2:1b'
    print(nomic.embed_words(["Bachelor", "Thesis"]))