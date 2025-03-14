import torch
from torch import Tensor

from text.Embedding.bert import Bert
from text.Embedding.deepseek import DeepSeek1B, DeepSeek7B
from text.Embedding.gpt2 import GPT2
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.nlp_adbench import NLP_ADBench
from text.dataset_converter.dataset_embedder import DatasetEmbedder
import random

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == '__main__':
    from transformers import pipeline, set_seed, TextGenerationPipeline

    generator: TextGenerationPipeline = pipeline('text-generation', model='meta-llama/Llama-3.2-1B', device=device)
    set_seed(42)
    dataset = EmotionDataset()
    #model = DeepSeek1B()
    samples, _ = dataset.get_training_data(filter_labels=dataset.get_possible_labels()[0:1])
    #classification_postfix = ". The emotion I am feeling is that of "
    classification_postfix = ". I am feeling"
    i = 0
    for sample in samples:
        #print(sample)
        #print(generator(sample, max_length=30, num_return_sequences=5))
        extended_sample = sample + classification_postfix
        #extended_sample = "The sample: \"" + sample + "\". expresses the emotion of"
        #sample_length = model.tokenize(extended_sample).shape[0]
        sample_length = generator.tokenizer(extended_sample, return_tensors='pt').data['input_ids'].shape[1]

        generated = generator(extended_sample, max_length=sample_length + 5, num_return_sequences=1)[0]['generated_text']
        #generated = generator(extended_sample, num_return_sequences=1)[0][ 'generated_text']
        last_word = " ".join(generated.split(" ")[-5])
        print(last_word + f"\n\t{generated}")
        if i > 10:
            break
        i += 1