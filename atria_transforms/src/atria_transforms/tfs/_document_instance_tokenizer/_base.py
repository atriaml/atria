from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from atria_logger import get_logger
from atria_types import DocumentInstance
from atria_types._data_instance._exceptions import AnnotationNotFoundError
from atria_types._generic._annotations import AnnotationType
from pydantic import Field

from atria_transforms.core import DataTransform
from atria_transforms.data_types._tokenized_document_instance import (
    PreTokenizedDocumentInstance,
)
from atria_transforms.registry import DATA_TRANSFORMS
from atria_transforms.tfs import HuggingfaceProcessor

from ._utilities import (
    _document_instance_to_hf_processor_inputs,
    _post_process_tokenizer_outputs,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@DATA_TRANSFORMS.register("document_instance_pre_tokenizer")
class DocumentInstancePreTokenizer(DataTransform[PreTokenizedDocumentInstance]):
    hf_processor: HuggingfaceProcessor = Field(default_factory=HuggingfaceProcessor)

    # segment-level-rank info args
    add_segment_level_info: bool = False
    use_segment_level_bboxes: bool = False
    max_segment_num: int = 150

    @property
    def data_model(self):
        return PreTokenizedDocumentInstance

    def __call__(
        self, document_instance: DocumentInstance
    ) -> PreTokenizedDocumentInstance | list[PreTokenizedDocumentInstance]:
        assert isinstance(document_instance, DocumentInstance), (
            f"{self.__class__.__name__} only supports DocumentInstance inputs, "
            f"but received input of type {type(document_instance)}"
        )

        # convert DocumentInstance to Huggingface processor inputs
        hf_processor_inputs = _document_instance_to_hf_processor_inputs(
            document_instance, use_segment_level_bboxes=self.use_segment_level_bboxes
        )

        # perform tokenization using the hf_processor
        tokenization_data = self.hf_processor(hf_processor_inputs)

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

        try:
            label = document_instance.get_annotation_by_type(
                AnnotationType.classification
            ).label.value
            processed_outputs["label"] = torch.tensor(label, dtype=torch.long)
        except AnnotationNotFoundError:
            pass

        # if image exists we ignore it and replace with the original one
        if "image" in processed_outputs:
            processed_outputs["image"] = document_instance.image

        # attach more info
        return PreTokenizedDocumentInstance(
            **{
                "index": document_instance.index,
                "sample_id": document_instance.sample_id,
                "words": hf_processor_inputs.pop("text", []),
                **processed_outputs,
            }
        )
