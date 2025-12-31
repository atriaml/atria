from __future__ import annotations

from atria_logger import get_logger
from atria_types import DocumentInstance
from atria_types._generic._annotations import (
    AnnotationType,
    QuestionAnsweringAnnotation,
)
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
    _post_process_tokenizer_outputs,
)

logger = get_logger(__name__)


@DATA_TRANSFORMS.register("document_pre_tokenizer")
class DocumentTokenizer(DataTransform[TokenizedDocumentInstance]):
    hf_processor: HuggingfaceProcessor = Field(default_factory=HuggingfaceProcessor)

    # segment-level-rank info args
    use_segment_level_bboxes: bool = False
    remove_no_answer_samples: bool = False
    image_size: tuple[int, int] | None = None

    @property
    def data_model(self):
        return TokenizedDocumentInstance

    def _default_call(self, document_instance: DocumentInstance):
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
            load_bboxes=True,
            load_image=False,
        )

        # remove image field if present
        processed_outputs.pop("image", None)

        # get imge and resize if needed
        image = document_instance.image
        if image is not None and self.image_size is not None:
            image = image.ops.resize(
                width=self.image_size[0], height=self.image_size[1]
            )

        # attach more info
        processed_outputs = {
            "index": document_instance.index,
            "sample_id": document_instance.sample_id,
            "words": hf_processor_inputs.pop("text", []),
            "image": image,
            "annotations": document_instance.annotations,
            **processed_outputs,
        }

        # attach more info
        return TokenizedDocumentInstance(**processed_outputs)

    def _qa_call(self, document_instance: DocumentInstance):
        qa_annotations = document_instance.get_annotation_by_type(
            annotation_type=AnnotationType.question_answering
        )

        document_instance_per_qa_pair = []
        for qa_pair in qa_annotations.qa_pairs:
            annotation = QuestionAnsweringAnnotation(qa_pairs=[qa_pair])

            def has_answer(qa_pair: QAPair):
                for answer_span in qa_pair.answer_spans:
                    if answer_span.start != -1 and answer_span.end != -1:
                        return True
                return False

            assert qa_pair.answer_spans is not None
            if self.remove_no_answer_samples:
                if not has_answer(qa_pair):
                    logger.debug(
                        f"Skipping QA Pair with id {qa_pair.id} due to no answer spans."
                    )
                    continue

            new_document_instance = document_instance.model_copy(
                update={
                    "sample_id": f"{document_instance.sample_id}_qa_{qa_pair.id}",
                    "annotations": [annotation],
                }
            )
            document_instance_per_qa_pair.append(new_document_instance)

        return document_instance_per_qa_pair

    def __call__(
        self, document_instance: DocumentInstance
    ) -> TokenizedDocumentInstance | list[TokenizedDocumentInstance]:
        if document_instance.has_annotation_type(
            annotation_type=AnnotationType.question_answering
        ):
            document_instances = self._qa_call(document_instance)
            processed_instances = []
            for doc_instance in document_instances:
                processed_instance = self._default_call(doc_instance)
                processed_instances.append(processed_instance)
            return processed_instances
        else:
            return self._default_call(document_instance)
