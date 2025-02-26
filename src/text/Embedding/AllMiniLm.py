import time

from sentence_transformers import SentenceTransformer

from text.Embedding.deepseek import DeepSeek1B
from text.dataset.emotions import EmotionDataset

if __name__ == '__main__':
    #sentences = ["This is an example sentence", "Each sentence is converted"]
    dataset = EmotionDataset()
    deepseek = DeepSeek1B()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to("cuda")
    i=0
    start = time.time()
    for sample in dataset.get_training_data()[0]:
        model.encode(sample)
        if i % 1000 == 0:
            print(f"embedded {i} sentences")
        i += 1
        if i == 5000:
            break
    print(f"time taken for MiniLM embedding {time.time() - start} seconds")
    start = time.time()
    i=0
    for sample in dataset.get_training_data()[0]:
        tokenized = deepseek.tokenize_batch([sample])
        emb = deepseek.get_embedding_fun(batch_first=True)(tokenized)
        if i % 1000 == 0:
            print(f"embedded {i} sentences")
        i += 1
        if i == 5000:
            break
    print(f"time taken for DeepSeek embedding {time.time() - start} seconds")
    start = time.time()
    i=0
