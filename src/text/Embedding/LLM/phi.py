from text.Embedding.LLM.causal_llm import CausalLLM


class Phi4B(CausalLLM):
    """
    This class is a wrapper for the Phi model.
    It uses the Hugging Face Transformers library to load and use the model.
    """

    @property
    def _model_prefix(self):
        return "microsoft/"

    @property
    def _model_name(self):
        return "Phi-4-mini-instruct"