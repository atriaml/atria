from __future__ import annotations

import numpy as np
from atria_logger import get_logger
from atria_types import DocumentInstance
from atria_types._generic._annotations import AnnotationType
from atria_types._generic._qa_pair import QAPair
from pydantic import Field

from atria_transforms.core import DataTransform
from atria_transforms.data_types._tokenized_document_instance import (
    TokenizedDocumentInstance,
)
from atria_transforms.registry import DATA_TRANSFORMS
from atria_transforms.tfs import HuggingfaceProcessor
from atria_transforms.tfs._utilities import (
    _document_instance_to_hf_processor_inputs,
    _generate_qa_token_ids,
    _post_process_tokenizer_outputs,
)

logger = get_logger(__name__)


@DATA_TRANSFORMS.register("document_tokenizer")
class DocumentTokenizer(DataTransform[TokenizedDocumentInstance]):
    hf_processor: HuggingfaceProcessor = Field(default_factory=HuggingfaceProcessor)

    # segment-level-rank info args
    use_segment_level_bboxes: bool = False
    remove_no_answer_samples: bool = False
    image_size: tuple[int, int] | None = None
    save_images: bool = True
    save_bboxes: bool = True

    @property
    def data_model(self):
        return TokenizedDocumentInstance

    def __call__(
        self, document_instance: DocumentInstance
    ) -> TokenizedDocumentInstance | list[TokenizedDocumentInstance]:
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

        # post-process tokenizer outputs
        processed_outputs = _post_process_tokenizer_outputs(
            tokenization_data=tokenization_data,
            input_word_boxes=hf_processor_inputs.get("boxes", None),
            input_word_labels=hf_processor_inputs.get("word_labels", None),
            input_image=hf_processor_inputs.get("images", None),
            all_special_ids=self.hf_processor.tokenizer.all_special_ids,
            load_bboxes=self.save_bboxes,
            load_image=False,
        )

        # attach more info
        processed_outputs = {
            "index": document_instance.index,
            "sample_id": document_instance.sample_id,
            "words": hf_processor_inputs.pop("text", []),
            "annotations": document_instance.annotations,
            **processed_outputs,
        }

        if self.save_images:
            # remove image field if present
            processed_outputs.pop("image", None)

            # get imge and resize if needed
            image = document_instance.image
            if image is not None and self.image_size is not None:
                image = image.ops.resize(
                    width=self.image_size[0], height=self.image_size[1]
                )
            processed_outputs["image"] = image

        # attach more info
        return TokenizedDocumentInstance(**processed_outputs)


@DATA_TRANSFORMS.register("document_tokenizer/question_answering")
class QuestionAnsweringDocumentTokenizer(DataTransform[TokenizedDocumentInstance]):
    hf_processor: HuggingfaceProcessor = Field(default_factory=HuggingfaceProcessor)

    # segment-level-rank info args
    use_segment_level_bboxes: bool = False
    ignore_no_answer_qa_pair: bool = False
    load_images: bool = True
    load_bboxes: bool = True

    @property
    def data_model(self):
        return TokenizedDocumentInstance

    def _tokenize_instance(
        self, document_instance: DocumentInstance, qa_pair: QAPair
    ) -> TokenizedDocumentInstance | None:
        # convert DocumentInstance to Huggingface processor inputs
        hf_processor_inputs = _document_instance_to_hf_processor_inputs(
            document_instance,
            use_segment_level_bboxes=self.use_segment_level_bboxes,
            context=qa_pair.question_text,
        )

        # perform tokenization using the hf_processor
        tokenization_data = self.hf_processor(hf_processor_inputs)

        return tokenization_data

    def _process_instance_for_qa_pair(
        self, document_instance: DocumentInstance
    ) -> TokenizedDocumentInstance | None:
        qa_annotation = document_instance.get_annotation_by_type(
            annotation_type=AnnotationType.question_answering
        )
        assert len(qa_annotation.qa_pairs) == 1, (
            "DocumentInstance passed to _process_instance_for_qa_pair "
            "must contain only one QAPair in its QuestionAnsweringAnnotation."
        )
        qa_pair = qa_annotation.qa_pairs[0]
        assert len(qa_pair.answer_spans) > 0, (
            f"QA Pair {qa_pair.id} has no answer spans."
        )

        # convert DocumentInstance to Huggingface processor inputs
        hf_processor_inputs = _document_instance_to_hf_processor_inputs(
            document_instance,
            use_segment_level_bboxes=self.use_segment_level_bboxes,
            context=qa_pair.question_text,
        )

        # perform tokenization using the hf_processor
        tokenization_data = self.hf_processor(hf_processor_inputs)

        # post-process tokenizer outputs
        processed_outputs = _post_process_tokenizer_outputs(
            tokenization_data=tokenization_data,
            input_word_boxes=hf_processor_inputs.get("boxes", None),
            input_word_labels=hf_processor_inputs.get("word_labels", None),
            input_image=hf_processor_inputs.get("images", None),
            all_special_ids=self.hf_processor.tokenizer.all_special_ids,
            load_bboxes=self.save_bboxes,
            load_image=False,
        )

        # generate token-level answer start and end positions
        token_answer_start, token_answer_end = _generate_qa_token_ids(
            qa_pair=qa_pair,
            word_ids=processed_outputs["word_ids"],
            sequence_ids=processed_outputs["sequence_ids"],
            sequence_length=processed_outputs["token_ids"].shape[-1],
        )

        # if all token_answer_start and token_answer_end are 0, it means we could not find the answer in the context
        # therefore using this sample as a training sample will not help the model learn anything
        if self.ignore_no_answer_qa_pair:
            total_answers = len(token_answer_start)
            for key, value in processed_outputs.items():
                if value is None:
                    continue
                if key == "image":
                    continue
                assert len(value) == total_answers, (
                    f"Length mismatch in tokenization data for key {key}. "
                    f"Expected length: {total_answers}, Actual length: {len(value)}"
                )

            valid_indices = []
            for idx, (s, e) in enumerate(
                zip(token_answer_start, token_answer_end, strict=True)
            ):
                if s != -1 and e != -1:
                    valid_indices.append(idx)

            if len(valid_indices) == 0:
                raise ValueError(
                    "All QA pairs in this document instance have no answer. "
                    "Consider setting ignore_no_answer_qa_pair to False."
                )

            if len(valid_indices) < total_answers:
                processed_outputs = {
                    k: v[valid_indices] if v is not None and k not in ["image"] else v
                    for k, v in processed_outputs.items()
                }
                token_answer_start = token_answer_start[valid_indices]
                token_answer_end = token_answer_end[valid_indices]

            assert (np.array(token_answer_end) != -1).all(), (
                f"Some end answer indices are -1 in document {token_answer_end}"
            )
            assert (np.array(token_answer_start) != -1).all(), (
                f"Some start answer indices are -1 in document {token_answer_start}"
            )
            total_answers = len(token_answer_start)
            for key, value in processed_outputs.items():
                if value is None:
                    continue
                if key == "image":
                    continue
                assert len(value) == total_answers, (
                    f"Length mismatch in tokenization data for key {key}. "
                    f"Expected length: {total_answers}, Actual length: {len(value)}"
                )

        # attach more info
        processed_outputs = {
            "index": document_instance.index,
            "sample_id": document_instance.sample_id,
            "words": hf_processor_inputs.pop("text", []),
            "annotations": document_instance.annotations,
            "token_answer_start": token_answer_start,
            "token_answer_end": token_answer_end,
            **processed_outputs,
        }

        if self.save_images:
            processed_outputs["image"] = image

        # attach more info
        return TokenizedDocumentInstance(**processed_outputs)

    def __call__(
        self, document_instance: DocumentInstance
    ) -> TokenizedDocumentInstance | list[TokenizedDocumentInstance]:
        document_instance_per_qa_pair = []
        for (
            qa_pair_document_instance
        ) in document_instance.ops.get_instances_per_qa_pair(
            ignore_no_answer_qa_pair=self.ignore_no_answer_qa_pair
        ):
            document_instance_per_qa_pair.append(
                self._process_instance_for_qa_pair(qa_pair_document_instance)
            )

        return document_instance_per_qa_pair
