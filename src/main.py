from typing import Tuple

import torch
from torch import Tensor

from text.Embedding.bert import Bert
from text.Embedding.deepseek import DeepSeek1B, DeepSeek7B
from text.Embedding.gpt2 import GPT2
from text.Embedding.llama import LLama1B, LLama3B
from text.dataset.ag_news import AGNews
from text.dataset.emotions import EmotionDataset
from text.dataset.imdb import IMBdDataset
from text.dataset.nlp_adbench import NLP_ADBench
from text.dataset_converter.dataset_embedder import DatasetEmbedder
import random

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == '__main__':
    from transformers import pipeline, set_seed, TextGenerationPipeline

    #generator: TextGenerationPipeline = pipeline('text-generation', model='meta-llama/Llama-3.2-1B', device=device)
    #model = LLama1B()
    set_seed(42)
    dataset = EmotionDataset()
    model = LLama1B()
    samples, _ = dataset.get_training_data(filter_labels=dataset.get_possible_labels()[0:1])
    #classification_postfix = ". The emotion I am feeling is that of "
    classification_postfix = ". feeling:"
    # Six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).
    classification_prefix = ("text: I am happy because it is sunny. feeling: not sadness\n"
                             "text: I feel sad because I have no friends. feeling: sadness\n"
                             "text: ")
    i = 0
    for sample in samples:
        #print(sample)
        #print(generator(sample, max_length=30, num_return_sequences=5))
        extended_sample = classification_prefix + sample + classification_postfix
        tokenized = model.tokenize(extended_sample).to(device)
        inputs_embeds = model.embed_tokenized(tokenized).to(device).unsqueeze(0)

        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        output = model.model(inputs_embeds=inputs_embeds, output_attentions=True)
        #print(output)
        attentions: Tuple[Tensor] = output.attentions # 16-Tuple of attention tensors of dim (1, 32, token_length, token_length)
        attention_tensor = torch.stack(attentions, dim=0) # (16, 32, token_length, token_length)
        # Get the mean of the attention tensors for the last token.
        mean_attention_last_layer = attention_tensor[-1].mean(dim=0).mean(dim=0)[-1]
        mean_attention = attention_tensor.mean(dim=0).mean(dim=0).mean(dim=0)[-1]
        #print(mean_attention)
        #print(mean_attention_last_layer)
        print("mean")
        token_list = tokenized.tolist()
        decoded = [model.detokenize([token]) for token in token_list]
        print(sample)
        for attention in [mean_attention, mean_attention_last_layer]:
            attention_list = attention.tolist()
            paired = [(attention, token) for token, attention in zip(decoded, attention_list)]
            paired.sort(key=lambda x: x[0], reverse=True)
            print(paired)
            print([p[1] for p in paired])
        print()

        #extended_sample = "The sample: \"" + sample + "\". expresses the emotion of"
        #sample_length = model.tokenize(extended_sample).shape[0]
        #sample_length = generator.tokenizer(extended_sample, return_tensors='pt').data['input_ids'].shape[1]

        #generated = generator(extended_sample, max_length=sample_length + 5, num_return_sequences=1)[0]['generated_text']
        #generated = generator(extended_sample, num_return_sequences=1)[0][ 'generated_text']
        #last_word = " ".join(generated.split(" ")[-5])
        #print(last_word + f"\n\t{generated}")
        #if i > 10:
        #    break
        #i += 1