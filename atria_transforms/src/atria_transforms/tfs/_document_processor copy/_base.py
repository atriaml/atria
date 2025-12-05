from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from atria_types import DocumentInstance
from pydantic import Field

from atria_transforms.core import DataTransform
from atria_transforms.data_types import DocumentTensorDataModel
from atria_transforms.registry import DATA_TRANSFORM
from atria_transforms.tfs import HuggingfaceProcessor, StandardImageTransform

from ._utilities import (
    _document_instance_to_hf_processor_inputs,
    _post_process_tokenizer_outputs,
)

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import BatchEncoding

logger = get_logger(__name__)


@DATA_TRANSFORM.register("document_processor")
class DocumentInstanceProcessor(DataTransform[DocumentTensorDataModel]):
    hf_processor: HuggingfaceProcessor = Field(default_factory=HuggingfaceProcessor)
    image_transform: StandardImageTransform = Field(
        default_factory=StandardImageTransform
    )

    # segment-level-rank info args
    add_segment_level_info: bool = False
    use_segment_level_bboxes: bool = False
    max_segment_num: int = 150

    def __call__(
        self, document_instance: DocumentInstance
    ) -> DocumentTensorDataModel | list[DocumentTensorDataModel]:
        # convert DocumentInstance to Huggingface processor inputs
        hf_processor_inputs = _document_instance_to_hf_processor_inputs(
            document_instance,
            use_segment_level_bboxes=self.use_segment_level_bboxes,
            image_transform=self.image_transform,
        )

        # perform tokenization using the hf_processor
        tokenization_data = self.hf_processor(**hf_processor_inputs)

        # post-process tokenizer outputs to generate segment-level info and align word ids and labels
        processed_outputs = self._post_process_tokenizer_outputs(
            document_instance=document_instance,
            hf_processor_inputs=hf_processor_inputs,
            tokenization_data=tokenization_data,
        )

        # for each token processed_outputs, create DocumentTensorDataModel
        bs = processed_outputs["token_ids"].shape[0]
        processed = [
            self._resolve_overflow(
                processed_outputs=processed_outputs,
                overflow_sample_idx=overflow_sample_idx,
            )
            for overflow_sample_idx in range(bs)
        ]
        if bs == 1:
            return processed[0]
        return processed

    def _post_process_tokenizer_outputs(
        self,
        document_instance: DocumentInstance,
        hf_processor_inputs: dict[str, Any],
        tokenization_data: BatchEncoding,
    ) -> dict[str, Any]:
        # post-process tokenizer outputs to generate segment-level info and align word ids and labels
        processed_outputs = _post_process_tokenizer_outputs(
            tokenization_data=tokenization_data,
            input_word_boxes=hf_processor_inputs.get("boxes", None),
            input_word_labels=hf_processor_inputs.get("word_labels", None),
            input_image=hf_processor_inputs.get("images", None),
            add_segment_level_info=self.add_segment_level_info,
            all_special_ids=self.hf_processor.tokenizer.all_special_ids,
            max_segment_num=self.max_segment_num,
        )

        # attach more info
        processed_outputs = {
            "index": document_instance.index,
            "sample_id": document_instance.sample_id,
            "words": hf_processor_inputs.pop("text", []),
            **processed_outputs,
        }

        return processed_outputs

    def _resolve_overflow(
        self, processed_outputs: dict[str, Any], overflow_sample_idx: int
    ) -> DocumentTensorDataModel:
        data = {
            key: value[overflow_sample_idx]
            for key, value in processed_outputs.items()
            if key
            in [
                "token_ids",
                "attention_mask",
                "token_bboxestoken_type_ids",
                "token_labels",
                "sequence_ids",
                "word_ids",
            ]
        }

        # image is the same for all overflowing segments of the same document
        if "image":
            data["image"] = processed_outputs["image"]

        # if label is present at sample level, add it
        if "label" in processed_outputs:
            data["label"] = processed_outputs["label"]

        # by default, we return for eery overflowing segment a separate DocumentTensorDataModel
        index = processed_outputs.get("index", None)
        sample_id = processed_outputs.get("sample_id", None)
        assert sample_id is not None, "sample_id must be present in processed outputs"
        words = processed_outputs.get("words", None)
        assert words is not None, "words must be present in processed outputs"
        return DocumentTensorDataModel(
            index=index,  # type: ignore
            sample_id=sample_id,  # type: ignore
            words=words,  # type: ignore
            **processed_outputs,
        )
