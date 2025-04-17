import unittest

import torch
from torch import Tensor

from text.Embedding.gpt2 import GPT2
from text.Embedding.LLM.llama import LLama3B
from text.Embedding.unification_strategy import UnificationStrategy

device = torch.device('cuda:0' if torch.cuda.is_available() else ('mps:0' if torch.backends.mps.is_available() else 'cpu'))

class HuggingModelTest(unittest.TestCase):

    def test_convert_mask(self):
        words = ["ive", "blabbed", "on"]
        mask = Tensor([0, 1, 1])
        hugging_model = GPT2()
        token_mask = hugging_model._convert_word_to_token_mask(words, mask)
        self.assertTrue(token_mask.equal(Tensor([0, 1, 1, 1, 1])))

    def test_convert_maintains_gradient(self):
        words = ["ive", "blabbed", "on"]
        mask = Tensor([0.0, 1.0, 1.0])
        mask.requires_grad = True
        mask = mask * 1.0
        hugging_model = GPT2()
        token_mask = hugging_model._convert_word_to_token_mask(words, mask)
        summed = token_mask.sum()
        summed.backward()
        # Set a breakpoint ot see if the functions are correctly maintaining the gradient.

    def test_aggregate_embeddings(self):
        #words = ["ive", "blabbed", "on"]
        embeddings = Tensor([[1, 1, 1], [1, 2, 3], [4, 5, 6], [7, 8, 9], [3, 3, 3]]).to(device)
        tokenized = [Tensor([1]), Tensor([1, 2, 3]), Tensor([4])]
        hugging_model = GPT2()
        aggregated = hugging_model._aggregate_token_to_word_embedding(embeddings, tokenized)
        expected = Tensor([[1, 1, 1], [4, 5, 6], [3, 3, 3]]).to(device)
        self.assertTrue(aggregated.equal(expected))

    def test_embed_mask(self):
        words = ["ive", "blabbed", "on"]
        mask = Tensor([0., 1., 1.]).to(device)
        hugging_model = GPT2()
        embedded = hugging_model.embed_words(words, mask)
        self.assertTrue(embedded.shape == (3, 768))
        self.assertEqual(embedded[0].sum(), 0)

    def test_embed_words_same_length(self):
        words = ['i', 'go', 'on', 'a', 'rant', 'about', 'my', 'insignificant', 'life', 'one', 'question', 'is', 'it', 'too', 'pretentious', 'to', 'say', 'i', 'feel']
        mask = None
        hugging_model = GPT2()
        embedded = hugging_model.embed_words(words, mask)
        self.assertTupleEqual(embedded.shape, (len(words), 768))

    def test_word_vs_token_embeddings(self):
        #samples = ["first example", "second example"]
        sample = "This is an example"
        model = LLama3B()
        tokenized = model.tokenize(sample).to(device).unsqueeze(0)

        #word_embs = model.embed_sentences(samples, strategy=UnificationStrategy.MEAN.create()).mean(1)
        word_embs = model.embed_sentences([sample], strategy=UnificationStrategy.MEAN.create()).mean(1)
        #tokenized = torch.stack([model.tokenize(sample).to(device) for sample in samples])
        embedding_fun = model.get_embedding_fun(batch_first=False)
        token_embs = embedding_fun(tokenized).T
        #self.assertListEqual(word_embs.tolist(), token_embs.tolist())
        self.assertTrue(torch.equal(word_embs, token_embs))


