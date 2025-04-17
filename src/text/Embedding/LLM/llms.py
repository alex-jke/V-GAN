from typing import List, Type

from text.Embedding.LLM.causal_llm import CausalLLM
from text.Embedding.LLM.deepseek import DeepSeek1B, DeepSeek7B
from text.Embedding.LLM.gemma import Gemma1BPt, Gemma4BPt, Gemma4BIt
from text.Embedding.LLM.llama import LLama1B, LLama3B, LLama3BInstruct
from text.Embedding.LLM.ministral import Ministral8BInstruct
from text.Embedding.LLM.open_elm import OpenELM1B, OpenELM3B, OpenELM3BInstruct
from text.Embedding.LLM.phi import Phi4B


def get_causal_llms() -> List[Type[CausalLLM]]:
    """
    Returns a list of all available Causal LLMs.
    Each LLM should be a subclass of CausalLLM.
    This function is used to dynamically load all available LLMs.
    Additionally, the models are returned as a list of classes, which can be used to instantiate them later.
    This is important, as when the models are instantiated, they will be loaded into GPU memory.
    """
    return [
        # Small DeepSeek model
        DeepSeek1B,

        # meta's LLama models
        LLama1B,
        LLama3B,
        LLama3BInstruct,

        # google's Gemma models
        Gemma1BPt,
        Gemma4BPt,
        Gemma4BIt,

        # microsoft's Phi mini instruct model
        Phi4B,

        # apples OpenELM models
        OpenELM1B,
        OpenELM3B,
        OpenELM3BInstruct,

        # Ministral
        Ministral8BInstruct,

        # Larger DeepSeek model
        DeepSeek7B

    ]