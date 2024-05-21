import gc
import logging
import os
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional, Dict, Generator

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from transformers.utils.quantization_config import QuantizationConfigMixin


@lru_cache(maxsize=1)
def get_embeddings_model():
    return HuggingFaceEmbeddings()


@lru_cache(maxsize=1)
def get_llm_model(
        model_id: str,
        hf_token: Optional[str] = None,
        quantization_int4: bool = True,
        **models_kwargs: Dict

) -> (AutoModelForCausalLM, AutoTokenizer):
    """
    Charge un modèle de langage de type CausalLM.

    Args:
        model_id (str): Identifiant du modèle à charger.
        hf_token (Optional[str]): Token d'accès à l'API Hugging Face.
        quantization_int4 (bool): Utilise la quantification 4-bit.
        models_kwargs (Optional[dict]): Arguments à passer au modèle.

    Returns:
        AutoModelForCausalLM: Modèle de langage.
        GemmaTokenizer: Tokenizer du modèle.
    """

    if hf_token is None:
        hf_token = os.environ["HF_TOKEN"]

    if models_kwargs is None:
        models_kwargs = {'max_length': 512}

    # Libère toute la mémoire allouée par le modèle de langage.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # https://huggingface.co/blog/4bit-transformers-bitsandbytes
    quantization_config = get_quantization_config(quantization_int4)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=quantization_config,
        **models_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    logging.debug(f'Modèle chargé : {model_id}')
    logging.debug(f'CausalLM : {model.__class__.__name__}')
    logging.debug(f'Tokenizer : {tokenizer.__class__.__name__}')

    return model, tokenizer


def get_quantization_config(quantization_int4: bool) -> Optional[QuantizationConfigMixin]:
    """
    Récupère la configuration de quantification demandée.

    Args:
        quantization_int4 (bool): Utilise la quantification 4-bit.

    Returns:
        Optional[BitsAndBytesConfig]: Configuration de quantification.
    """
    if quantization_int4:
        logging.debug("Utilisation de la quantification 4-bit.")

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    logging.debug("Aucune quantification n'est utilisée.")
    return None


@contextmanager
def log_inference(model_id: str) -> Generator[None, None, None]:
    """ Décorateur pour logger l'exécution d'un modèle. """

    logging.debug(f"Running model '{model_id}' on thinking")
    time_start = time.time()

    yield

    time_end = time.time()
    logging.debug(f"Model '{model_id}' inference took {time_end - time_start:.2f}.")
