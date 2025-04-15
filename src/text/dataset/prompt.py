from typing import List


class Prompt:
    """
    A class to create a prompt for a dataset. It takes a sample prefix, label prefix, samples, and labels.
    It creates a prefix and suffix for the prompt.
    """
    def __init__(self,
                 sample_prefix: str,
                 label_prefix: str,
                 samples: List[str],
                 labels: List[str],
                 instruction: str = ""):
        if len(samples) != len(labels):
            raise ValueError("Samples and labels must have the same length.")


        prefix = [" ".join([sample_prefix,sample, label_prefix, label, "\n"]) for sample,label in zip(samples, labels)]
        prefix.append(sample_prefix)
        prefix = [instruction] + prefix
        self.prefix = " ".join(prefix).split(" ")
        self.suffix = label_prefix.split(" ")
        self.full_prompt = " ".join(self.prefix +["[sample]"] +  self.suffix)

    def __repr__(self):
        return f"Prompt(prefix={self.prefix}, suffix={self.suffix})"