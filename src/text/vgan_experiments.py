import os

from text.Embedding.gpt2 import GPT2
from text.dataset.ag_news import AGNews
from text.dataset_converter.dataset_embedder import DatasetEmbedder
from text.dataset_converter.dataset_tokenizer import DatasetTokenizer
from text.visualizer.collective_visualizer import CollectiveVisualizer
from vgan import VGAN

if __name__ == "__main__":
    dataset = AGNews()
    labels = dataset.get_possible_labels()[:1]
    model = GPT2()
    embedder = DatasetEmbedder(model=model, dataset=dataset)
    embedded = embedder.embed(train=True, samples=-1, labels=labels)[0]
    epochs = 10000

    export_path = os.path.join(
        os.getcwd(),
        'experiments',
        'vgan'
    )
    vgan = VGAN(epochs=epochs,path_to_directory=export_path, lr_G=0.05)
    vgan.fit(embedded)

    visualizer = CollectiveVisualizer(tokenized_data=embedded,
                                      tokenizer=model,
                                      vmmd_model=vgan,
                                      export_path=export_path,
                                      text_visualization=False)
    visualizer.visualize(epoch=epochs, samples=30)