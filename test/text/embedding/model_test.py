import unittest

from text.Embedding.deepseek import DeepSeek, DeepSeek1B


class DeepSeekTest(unittest.TestCase):

    def test_embedding_padding(self):
        model = DeepSeek1B()
        tokenized = model.tokenize_batch(["Hello World"])
        print("number tokens:", tokenized.shape[1])
        embedding_fun = model.get_embedding_fun(batch_first=True)
        embedding = embedding_fun(tokenized)

        tokenized[0,0] = model.padding_token
        embedding_padded = embedding_fun(tokenized)
