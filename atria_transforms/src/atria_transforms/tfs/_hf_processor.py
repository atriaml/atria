from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from atria_logger import get_logger

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import (
        BatchEncoding,
        PreTrainedTokenizerBase,
    )

from atria_transforms.core import DataTransform
from atria_transforms.registry import DATA_TRANSFORMS

logger = get_logger(__name__)


@DATA_TRANSFORMS.register("hf_processor")
class HuggingfaceProcessor(DataTransform):
    tokenizer_name: str = "microsoft/layoutlmv3-base"

    # tokenizer config
    # file args
    cache_dir: str = "./cache"
    local_files_only: bool = False

    # ocr args only for multimodal processors
    apply_ocr: bool = False

    # image args only for multimodal processors
    do_normalize: bool = False
    do_resize: bool = False
    do_rescale: bool = False

    # text input processing args
    add_prefix_space: bool = True
    do_lower_case: bool = True

    add_special_tokens: bool = True
    padding: str | bool = "max_length"
    truncation: bool = True
    max_length: int = 512
    stride: int = 0
    pad_to_multiple_of: int = 8
    is_split_into_words: bool = True
    return_token_type_ids: bool | None = None
    return_attention_mask: bool = True
    return_special_tokens_mask: bool = False
    return_offsets_mapping: bool = False
    return_length: bool = False
    return_tensors: str = "pt"
    verbose: bool = True
    overflow_sampling: str = "return_all"

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return (
            self._hf_processor.tokenizer
            if hasattr(self._hf_processor, "tokenizer")
            else self._hf_processor
        )

    @property
    def all_special_ids(self) -> set[int]:
        return set(self.tokenizer.all_special_ids)

    def model_post_init(self, context) -> None:
        assert self.overflow_sampling in [
            "return_all",
            "return_random_n",
            "no_overflow",
            "return_first_n",
        ], f"Overflow sampling strategy {self.overflow_sampling} is not supported."

        self._hf_processor = None

    def _prepare_init_kwargs(self) -> dict:
        init_kwargs = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "apply_ocr": self.apply_ocr,
            "do_normalize": self.do_normalize,
            "do_resize": self.do_resize,
            "do_rescale": self.do_rescale,
            "add_prefix_space": self.add_prefix_space,
            "do_lower_case": self.do_lower_case,
        }
        return init_kwargs

    def _prepare_call_kwargs(self, processor: AutoProcessor) -> dict:
        call_kwargs = {
            "add_special_tokens": self.add_special_tokens,
            "padding": self.padding,
            "truncation": self.truncation,
            "max_length": self.max_length,
            "stride": self.stride,
            "pad_to_multiple_of": self.pad_to_multiple_of,
            "is_split_into_words": self.is_split_into_words,
            "return_overflowing_tokens": self.overflow_sampling != "no_overflow",
            "return_token_type_ids": self.return_token_type_ids,
            "return_attention_mask": self.return_attention_mask,
            "return_special_tokens_mask": self.return_special_tokens_mask,
            "return_offsets_mapping": self.return_offsets_mapping,
            "return_length": self.return_length,
            "return_tensors": self.return_tensors,
            "verbose": self.verbose,
        }

        self._possible_args = inspect.signature(processor.__call__).parameters  # type: ignore
        for key in list(call_kwargs.keys()):
            if key not in self._possible_args:
                logger.warning(
                    f"Invalid keyword argument '{key}' found in call_kwargs for {self.__class__.__name__}. Skipping it."
                )
                call_kwargs.pop(key)

        return call_kwargs

    def _initialize_transform(self):
        processor = AutoProcessor.from_pretrained(
            self.tokenizer_name, **self._prepare_init_kwargs()
        )
        self._call_kwargs = self._prepare_call_kwargs(processor)
        return processor

    def _convert_text_to_list(self, text: Any) -> list[str]:
        if isinstance(text, str):
            return text.split()
        elif isinstance(text, list):
            return text
        else:
            raise ValueError("Input text must be a string or a list of strings.")

    def __call__(self, input: dict) -> BatchEncoding:
        from transformers import BertTokenizerFast, RobertaTokenizerFast

        if not self._hf_processor:
            self._hf_processor = self._initialize_transform()

        filtered_inputs = {k: v for k, v in input.items() if k in self._possible_args}
        if isinstance(self.tokenizer, (BertTokenizerFast, RobertaTokenizerFast)):
            text = input.get("text", None)
            text_pair = input.get("text_pair", None)

            if text is not None and text_pair is not None:
                filtered_inputs["text"] = self._convert_text_to_list(text)
                filtered_inputs["text_pair"] = self._convert_text_to_list(text_pair)

                assert isinstance(filtered_inputs["text"], list), (
                    "Input 'text' must be a list of strings."
                )
                assert isinstance(input["text_pair"], list), (
                    "Input 'text_pair' must be a list of strings."
                )
        return self._hf_processor(**filtered_inputs, **self._call_kwargs)
