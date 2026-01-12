"""
models.py: Model and embedder loading utilities for FACTER.
"""
from __future__ import annotations

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from .config import Config

logger = logging.getLogger(__name__)


def load_embedder(prefer_public_finetuned: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = Config.EMBEDDER_ALT_PUBLIC if prefer_public_finetuned
    logger.info(f"Loading embedder: {name}")
    return SentenceTransformer(name).to(device)


def load_llm():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading LLM: {Config.LLM_BACKBONE}")
    tokenizer = AutoTokenizer.from_pretrained(Config.LLM_BACKBONE, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        Config.LLM_BACKBONE,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


def load_models(prefer_public_finetuned_embedder: bool = False):
    embedder = load_embedder(prefer_public_finetuned=prefer_public_finetuned_embedder)
    tokenizer, model = load_llm()
    return embedder, tokenizer, model
