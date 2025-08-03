"""
models.py: Model and embedder loading utilities for FACTER.
"""
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from .config import Config

logger = logging.getLogger(__name__)

def load_models():
    """
    Load the SentenceTransformer embedder and the LLM for recommendation.
    Used in offline + online calibration phases.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        logger.info("Loading embedding model...")
        embedder = SentenceTransformer('paraphrase-mpnet-base-v2').to(device)
        logger.info("Loading LLM...")
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B')
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            'meta-llama/Meta-Llama-3.1-8B',
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        return embedder, tokenizer, model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise
