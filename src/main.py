from typing import Tuple

import torch
from torch import Tensor

from text.Embedding.LLM.llama import LLama1B
from text.dataset.emotions import EmotionDataset
from text.dataset.nlp_adbench import NLP_ADBench
import random

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def gen_test():
    from transformers import pipeline, set_seed, TextGenerationPipeline

    generator: TextGenerationPipeline = pipeline('text-generation', model='meta-llama/Llama-3.2-1B', device=device)
    model = LLama1B()
    set_seed(42)
    dataset = EmotionDataset()
    model = LLama1B()
    samples, _ = dataset.get_training_data(filter_labels=dataset.get_possible_labels()[0:1])
    # classification_postfix = ". The emotion I am feeling is that of "
    classification_postfix = ". feeling:"
    # Six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).
    classification_prefix = ("text: I am happy because it is sunny. feeling: not sadness\n"
                             "text: I feel sad because I have no friends. feeling: sadness\n"
                             "text: ")
    i = 0
    for sample in samples:
        # print(sample)
        # print(generator(sample, max_length=30, num_return_sequences=5))
        extended_sample = classification_prefix + sample + classification_postfix
        tokenized = model.tokenize(extended_sample).to(device)
        inputs_embeds = model.embed_tokenized(tokenized).to(device).unsqueeze(0)

        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        output = model.model(inputs_embeds=inputs_embeds, output_attentions=True)
        # print(output)
        attentions: Tuple[
            Tensor] = output.attentions  # 16-Tuple of attention tensors of dim (1, 32, token_length, token_length)
        attention_tensor = torch.stack(attentions, dim=0)  # (16, 32, token_length, token_length)
        # Get the mean of the attention tensors for the last token.
        mean_attention_last_layer = attention_tensor[-1].mean(dim=0).mean(dim=0)[-1]
        mean_attention = attention_tensor.mean(dim=0).mean(dim=0).mean(dim=0)[-1]
        # print(mean_attention)
        # print(mean_attention_last_layer)
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

        # extended_sample = "The sample: \"" + sample + "\". expresses the emotion of"
        # sample_length = model.tokenize(extended_sample).shape[0]
        # sample_length = generator.tokenizer(extended_sample, return_tensors='pt').data['input_ids'].shape[1]

        generated = generator(extended_sample, max_length=sample_length + 5, num_return_sequences=1)[0][
            'generated_text']
        generated = generator(extended_sample, num_return_sequences=1)[0]['generated_text']
        # last_word = " ".join(generated.split(" ")[-5])
        # print(last_word + f"\n\t{generated}")
        # if i > 10:
        #    break
        # i += 1
def print_dataset_stats():
    datasets = NLP_ADBench.get_all_datasets()
    test_size = 2000
    train_size = 5000
    for dataset in datasets:
        train_data = dataset.get_training_data(filter_labels=[0])[0][:train_size]
        test_data = dataset.get_testing_data()[0][:test_size]
        length = lambda s: len(s.split(" "))
        total_train_length = train_data.apply(length).sum()
        total_test_length = test_data.apply(length).sum()
        avg_test_length = total_test_length / test_size
        avg_train_length = total_train_length / train_size
        avg_total_length = (total_test_length + total_train_length) / (train_size + test_size)
        print(f"dataset: {dataset.name}, avg_train: {avg_train_length}, avg_test: {avg_test_length}, total_avg: {avg_total_length}")
"""
Char lengths:
dataset: NLP_ADBench agnews, avg_train: 189.2022, avg_test: 186.439, total_avg: 188.4127142857143
dataset: NLP_ADBench N24News, avg_train: 4825.1024, avg_test: 4583.6935, total_avg: 4756.128428571428
dataset: NLP_ADBench bbc, avg_train: 547.4994, avg_test: 663.9685, total_avg: 580.7762857142857
dataset: NLP_ADBench email_spam, avg_train: 450.4242, avg_test: 546.1975, total_avg: 477.788
dataset: NLP_ADBench emotion, avg_train: 97.4148, avg_test: 97.421, total_avg: 97.41657142857143
dataset: NLP_ADBench movie_review, avg_train: 1304.2906, avg_test: 1293.0405, total_avg: 1301.0762857142856
dataset: NLP_ADBench sms_spam, avg_train: 44.0234, avg_test: 59.396, total_avg: 48.415571428571425
dataset: NLP_ADBench yelp_review_polarity, avg_train: 623.8692, avg_test: 654.392, total_avg: 632.59
"""

"""
Word lengths:
dataset: NLP_ADBench agnews, avg_train: 31.48, avg_test: 31.013, total_avg: 31.34657142857143
dataset: NLP_ADBench N24News, avg_train: 817.7538, avg_test: 776.2575, total_avg: 805.8977142857143
dataset: NLP_ADBench bbc, avg_train: 93.9606, avg_test: 113.8205, total_avg: 99.63485714285714
dataset: NLP_ADBench email_spam, avg_train: 98.0288, avg_test: 117.022, total_avg: 103.45542857142857
dataset: NLP_ADBench emotion, avg_train: 19.2752, avg_test: 19.271, total_avg: 19.274
dataset: NLP_ADBench movie_review, avg_train: 232.5054, avg_test: 230.75, total_avg: 232.00385714285713
dataset: NLP_ADBench sms_spam, avg_train: 8.9418, avg_test: 11.8235, total_avg: 9.765142857142857
dataset: NLP_ADBench yelp_review_polarity, avg_train: 116.712, avg_test: 122.6705, total_avg: 118.41442857142857
"""
if __name__ == '__main__':
    print_dataset_stats()