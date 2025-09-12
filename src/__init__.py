import time
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, logging
from huggingface_hub import hf_hub_download
import os

# Suppress unnecessary warnings from the transformers library
logging.set_verbosity_error()


class SmallLLMModel:
    """
    A utility class wrapping a lightweight Hugging Face causal-LM for fast, low-memory experimentation.

    Parameters
    ----------
    model_name: str, default="Qwen/Qwen3-0.6B"
        Identifier of the model on the 🤗 Hub. Replace it with another model identifier if needed.
    device: str | None, default=None
        The computation device. If *None*, the device is automatically selected:
        - "mps" for macOS with Metal Performance Shaders
        - "cuda" for GPUs
        - "cpu" as a fallback
    dtype: torch.dtype | None, default=None
        Numerical precision. Defaults to:
        - `float16` for "cuda" or "mps" to reduce memory usage
        - `float32` for "cpu" for compatibility
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name

        # Automatically select the computation device
        self.device = self._select_device(device)

        # Set the numerical precision
        self.dtype = dtype or (torch.float16 if self.device in ["cuda", "mps"] else torch.float32)

        # Load the tokenizer and model
        self.tokenizer: PreTrainedTokenizer = self._load_tokenizer(model_name, trust_remote_code)
        self.model: PreTrainedModel = self._load_model(model_name, trust_remote_code)

    def _select_device(self, device: str | None) -> str:
        """
        Select the computation device based on availability.
        """
        if device:
            return device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_tokenizer(self, model_name: str, trust_remote_code: bool) -> PreTrainedTokenizer:
        """
        Load the tokenizer for the specified model.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token_id is None:
            # Ensure a pad token exists for batch processing
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _load_model(self, model_name: str, trust_remote_code: bool) -> PreTrainedModel:
        """
        Load the model for the specified model name.
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        model.to(self.device).eval()

        # Set the model to inference-only mode
        for param in model.parameters():
            param.requires_grad = False
        return model

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _encode(self, text: str) -> torch.Tensor:
        """
        Tokenize the input text and return a 2-D tensor of input IDs on the target device.
        """
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([input_ids], device=self.device, dtype=torch.long)

    def _decode(self, token_ids: torch.Tensor | list[int]) -> str:
        """
        Decode token IDs back into text, skipping special tokens.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # -------------------------------------------------------------------------
    # Public helpers
    # -------------------------------------------------------------------------

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """
        Compute the raw logits (no softmax) for the next token given a list of input token IDs.

        Parameters
        ----------
        input_ids: list[int]
            A list of token IDs representing the input sequence.

        Returns
        -------
        list[float]
            The logits for the next token in the sequence.
        """
        input_tensor = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        with torch.no_grad():
            output = self.model(input_ids=input_tensor)
        # Extract logits for the last token in the sequence
        logits = output.logits[0, -1].tolist()
        return [float(logit) for logit in logits]

    def get_path_to_vocabulary_json(self) -> str:
        """
        Download and return the path to the vocabulary JSON file for the tokenizer.

        Returns
        -------
        str
            The local path to the vocabulary JSON file.
        """
        vocab_file_name = self.tokenizer.vocab_files_names.get("vocab_file", "vocab.json")
        vocab_path = hf_hub_download(repo_id=self.model_name, filename=vocab_file_name)
        return vocab_path