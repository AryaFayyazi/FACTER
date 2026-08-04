"""
models.py: Model and embedder loading utilities for FACTER.
"""
from __future__ import annotations

import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from .config import Config

logger = logging.getLogger(__name__)


def load_embedder(prefer_public_finetuned: bool = True):
    """Load the sentence encoder used for every semantic distance in FACTER.

    (Earlier releases had an incomplete conditional expression on this line,
    which made the module raise SyntaxError on import.)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Config.EMBEDDER_ALT_PUBLIC if prefer_public_finetuned else Config.EMBEDDER_NAME
    logger.info(f"Loading embedder: {name}")
    return SentenceTransformer(name).to(device)


def load_llm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading LLM: {Config.LLM_BACKBONE}")
    name = os.environ.get("FACTER_LLM", Config.LLM_BACKBONE)
    logger.info(f"Loading LLM: {name}")
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


def load_models(prefer_public_finetuned_embedder: bool = False):
    embedder = load_embedder(prefer_public_finetuned=prefer_public_finetuned_embedder)
    tokenizer, model = load_llm()
    return embedder, tokenizer, model
