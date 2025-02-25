from sentence_transformers import SentenceTransformer

from text.dataset.emotions import EmotionDataset

if __name__ == '__main__':
    #sentences = ["This is an example sentence", "Each sentence is converted"]
    dataset = EmotionDataset()

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to("mps")
    i=0
    for sample in dataset.get_training_data()[0]:
        model.encode(sample)
        if i % 1000 == 0:
            print(f"embedded {i} sentences")
        i += 1
