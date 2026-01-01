from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Self

import torch
from atria_logger import get_logger
from pydantic import BaseModel, ConfigDict, model_validator

from atria_transforms.tfs._utilities import (
    _extract_sequence_and_word_ids,
    _extract_token_bboxes_from_word_bboxes,
    _extract_token_labels_from_word_labels,
)

if TYPE_CHECKING:
    from transformers import AutoProcessor

from atria_transforms.constants import _DEFAULT_ATRIA_TFS_CACHE_DIR
from atria_transforms.core import DataTransform
from atria_transforms.registry import DATA_TRANSFORMS

logger = get_logger(__name__)


class HuggingfaceProcessorInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")
    text: str | list[str]
    text_pair: list[str] | None = None
    boxes: list[list[float]] | None = None
    label: int | None = None
    word_labels: list[int] | None = None


class HuggingfaceProcessorOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")
    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor | None = None
    special_tokens_mask: torch.Tensor | None = None
    offsets_mapping: torch.Tensor | None = None
    sequence_ids: torch.Tensor
    word_ids: torch.Tensor
    token_bboxes: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    label: torch.Tensor | None = None

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        for field_name in self.__class__.model_fields.keys():
            field_value = getattr(self, field_name)
            if field_value is not None:
                assert field_value.shape[0] == self.batch_size, (
                    f"Batch size mismatch for field '{field_name}': "
                    f"expected {self.batch_size}, got {field_value.shape[0]}"
                )
                if field_name != "label":
                    assert field_value.shape[1] == self.sequence_length, (
                        f"Batch size mismatch for field '{field_name}': "
                        f"expected {self.batch_size}, got {field_value.shape[0]}"
                    )

        return self

    @property
    def batch_size(self) -> int:
        return self.token_ids.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.token_ids.shape[1]

    def take(self, indices: list[int]) -> HuggingfaceProcessorOutput:
        taken_data = {}
        for field_name, field_value in self.model_dump().items():
            if field_value is not None:
                taken_data[field_name] = field_value[indices]
            else:
                taken_data[field_name] = None
        return HuggingfaceProcessorOutput.model_validate(taken_data)


@DATA_TRANSFORMS.register("hf_processor")
class HuggingfaceProcessor(DataTransform):
    tokenizer_name: str = "microsoft/layoutlmv3-base"

    # tokenizer config
    # file args
    cache_dir: str | None = None
    local_files_only: bool = False

    # ocr args only for multimodal processors
    apply_ocr: bool = False

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

    @property
    def tokenizer(self) -> Any:
        return (
            self._hf_processor.tokenizer
            if hasattr(self._hf_processor, "tokenizer")
            else self._hf_processor
        )

    @property
    def all_special_ids(self) -> set[int]:
        return set(self.tokenizer.all_special_ids)

    def model_post_init(self, context) -> None:
        self._hf_processor = None

    def _prepare_init_kwargs(self) -> dict:
        init_kwargs = {
            "cache_dir": self.cache_dir or _DEFAULT_ATRIA_TFS_CACHE_DIR,
            "local_files_only": self.local_files_only,
            "apply_ocr": self.apply_ocr,
            "do_normalize": False,
            "do_resize": False,
            "do_rescale": False,
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
            "return_overflowing_tokens": True,
            "return_token_type_ids": self.return_token_type_ids,
            "return_attention_mask": self.return_attention_mask,
            "return_special_tokens_mask": self.return_special_tokens_mask,
            "return_offsets_mapping": self.return_offsets_mapping,
            "return_length": self.return_length,
            "return_tensors": self.return_tensors,
            "verbose": self.verbose,
        }

        self._possible_args = inspect.signature(processor.__call__).parameters.keys()  # type: ignore
        for key in list(call_kwargs.keys()):
            if key not in self._possible_args:
                logger.warning(
                    f"Invalid keyword argument '{key}' found in call_kwargs for {self.__class__.__name__}. Skipping it."
                )
                call_kwargs.pop(key)

        return call_kwargs

    def _initialize_transform(self):
        from transformers import AutoTokenizer

        processor = AutoTokenizer.from_pretrained(
            self.tokenizer_name, **self._prepare_init_kwargs()
        )
        self._call_kwargs = self._prepare_call_kwargs(processor)
        return processor

    def __call__(self, input: HuggingfaceProcessorInput) -> HuggingfaceProcessorOutput:
        from transformers import BertTokenizerFast, RobertaTokenizerFast

        if not self._hf_processor:
            self._hf_processor = self._initialize_transform()

        filtered_inputs = {
            k: v for k, v in input.model_dump().items() if k in self._possible_args
        }

        if isinstance(self.tokenizer, (BertTokenizerFast, RobertaTokenizerFast)):
            # for some reason Bert and Roberta tokenizers require text to be a list of strings
            # when a pair is passed, but layoutlm does its own logic, what a freakshow of a library
            # every single tokenizer has its own ghost quirks happening under the hood
            text = filtered_inputs.get("text", None)
            text_pair = filtered_inputs.get("text_pair", None)

            if text is not None and text_pair is not None:
                filtered_inputs["text"] = [text]

        tokenization_data = self._hf_processor(**filtered_inputs, **self._call_kwargs)

        # extract sequence_ids and word_ids
        sequence_ids, word_ids = _extract_sequence_and_word_ids(tokenization_data)  # noqa: F821

        # extract token_bboxes if needed, we always reextract to ensure alignment with word_ids
        token_bboxes = tokenization_data.get("bbox", None)
        if token_bboxes is None and input.boxes is not None:
            token_bboxes = _extract_token_bboxes_from_word_bboxes(input.boxes, word_ids)

        # extract token_labels if needed, we always reextract to ensure alignment with word_ids
        token_labels = tokenization_data.get("labels", None)
        if token_labels is None and input.word_labels is not None:
            token_labels = _extract_token_labels_from_word_labels(
                input.word_labels, word_ids
            )

        # extract label if needed
        label = None
        if input.label is not None:
            batch_size = tokenization_data["input_ids"].shape[0]
            label = torch.tensor([input.label] * batch_size, dtype=torch.long)

        return HuggingfaceProcessorOutput(
            token_ids=tokenization_data["input_ids"],
            attention_mask=tokenization_data["attention_mask"],
            token_type_ids=tokenization_data.get("token_type_ids"),
            special_tokens_mask=tokenization_data.get("special_tokens_mask"),
            offsets_mapping=tokenization_data.get("offset_mapping"),
            sequence_ids=sequence_ids,
            word_ids=word_ids,
            token_bboxes=token_bboxes if input.boxes is not None else None,
            token_labels=token_labels,
            label=label,
        )
