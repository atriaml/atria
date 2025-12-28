from __future__ import annotations

import torch
from atria_logger import get_logger
from huggingface_hub import hf_hub_download
from torch import nn

logger = get_logger(__name__)


class CheckpointLoader:
    """Handles loading and mapping of pretrained model checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        cache_dir: str,
        checkpoint_key_mapping: list[tuple[str, str]] | None = None,
        remove_root_prefix: bool = True,
    ):
        self.model = model
        self.cache_dir = cache_dir
        self.remove_root_prefix = remove_root_prefix
        self._rewrite_rules = checkpoint_key_mapping or []

    def load_from_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load checkpoint from local path."""
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        if checkpoint_path.startswith("hf://"):
            repo_id = checkpoint_path[5:]  # Remove 'hf://' prefix
            checkpoint_path = hf_hub_download(
                repo_id=repo_id, filename="pytorch_model.bin", cache_dir=self.cache_dir
            )

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        processed_state_dict = self._process_state_dict(state_dict)
        keys = self.model.load_state_dict(processed_state_dict, strict=strict)
        if strict:
            assert len(keys.missing_keys) == 0, f"Missing keys: {keys.missing_keys}"
            assert len(keys.unexpected_keys) == 0, (
                f"Unexpected keys: {keys.unexpected_keys}"
            )
        else:
            if len(keys.missing_keys) > 0:
                logger.warning(f"Warning: Missing keys: {keys.missing_keys}")
            if len(keys.unexpected_keys) > 0:
                logger.warning(f"Warning: Unexpected keys: {keys.unexpected_keys}")

    def _process_state_dict(self, state_dict: dict) -> dict:
        """Process raw state dict by removing prefix, remapping keys, and privatizing."""
        processed_dict = {}

        def _remove_root_prefix(key: str) -> str:
            """Remove the first component from a dot-separated key."""
            return ".".join(key.split(".")[1:])

        def _remap_key(key: str) -> str:
            """Apply rewrite rules to remap key names."""
            for old, new in self._rewrite_rules:
                if old in key:
                    key = key.replace(old, new)
            return key

        for key, value in state_dict.items():
            processed_dict[_remap_key(key)] = value

        processed_dict = {
            _remove_root_prefix(k) if self.remove_root_prefix else k: v
            for k, v in processed_dict.items()
        }

        return processed_dict
