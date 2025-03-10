from typing import Callable, Optional

import pandas as pd
import numpy as np
from numpy import ndarray
from torch import Tensor

from modules.text.vmmd_text_base import VMMDTextBase


class VmmdText(VMMDTextBase):
    """
    An implementation of VMMDTextBase that embeds the data each epoch.
    """
    def _get_training_data(self, x_data: ndarray[str], embedding: Callable[[ndarray[str], int], Tensor], n_dims: int) -> Tensor | ndarray[str]:
        self._n_dims = n_dims
        return x_data

    def _convert_batch(self, batch: ndarray[str] | Tensor, embedding: Callable[[ndarray[str], int], Tensor], mask: Optional[Tensor]) -> Tensor:
        """
        Converts a batch to an embedding tensor.
        """
        if mask is None:
            return embedding(batch, self._n_dims)
        df = pd.DataFrame(batch)
        words_df = df.apply(lambda s: self._sentence_to_words(s))
        sentences = words_df.values.tolist()
        masked_sentences = []

        for i in range(len(sentences)):
            sentence = sentences[i]
            mask = mask[i]
            new_sentence = ""
            for j in range(len(sentence)):
                if mask[j] == 1.0:
                    new_sentence += sentence[j]
            masked_sentences.append(new_sentence)

        return embedding(np.array(masked_sentences), self._n_dims)
