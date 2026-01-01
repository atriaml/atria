from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from atria_logger import get_logger
from atria_types import DocumentInstance

from atria_transforms.core import DataTransform
from atria_transforms.data_types import DocumentTensorDataModel
from atria_transforms.data_types._tokenized_document_instance import (
    TokenizedDocumentInstance,
)
from atria_transforms.registry.registry_groups import DATA_TRANSFORMS
from atria_transforms.tfs import StandardImageTransform
from atria_transforms.tfs._document_tokenizer import (
    DocumentTokenizer,
    QuestionAnsweringDocumentTokenizer,
    SequenceClassificationDocumentTokenizer,
    TokenClassificationDocumentTokenizer,
)
from atria_transforms.tfs._hf_processor import HuggingfaceProcessor

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class DocumentProcessor(DataTransform[DocumentTensorDataModel]):
    image_transform: StandardImageTransform = StandardImageTransform()

    # overflow sampling
    overflow_strategy: Literal["return_first", "return_random", "return_all"] = (
        "return_all"
    )

    def model_post_init(self, context: Any) -> None:
        self._document_tokenizer = self._initialize_document_tokenizer()

    @abstractmethod
    def _initialize_document_tokenizer(self) -> DocumentTokenizer:
        raise NotImplementedError

    def __call__(
        self, document_instance: DocumentInstance | TokenizedDocumentInstance
    ) -> DocumentTensorDataModel | list[DocumentTensorDataModel]:
        assert isinstance(
            document_instance, DocumentInstance | TokenizedDocumentInstance
        ), (
            f"{self.__class__.__name__} only supports DocumentInstance inputs, "
            f"but received input of type {type(document_instance)}"
        )

        if isinstance(document_instance, DocumentInstance):
            tokenized_instance = self._document_tokenizer(document_instance)
        else:
            tokenized_instance = document_instance

        if isinstance(tokenized_instance, TokenizedDocumentInstance):
            overflowed_instances = self._resolve_overflow_for_instance(
                tokenized_instance
            )
        elif isinstance(tokenized_instance, list):
            overflowed_instances = []
            for instance in tokenized_instance:
                overflowed_instances.extend(
                    self._resolve_overflow_for_instance(instance)
                )
        else:
            raise ValueError(
                "Tokenizer output must be a TokenizedDocumentInstance or a list of them."
            )

        return [
            DocumentTensorDataModel.from_tokenized_instance(
                tokenized_instance=instance, image_transform=self.image_transform
            )
            for instance in overflowed_instances
        ]

    def _resolve_overflow_for_instance(
        self, tokenized_instance: TokenizedDocumentInstance
    ) -> list[TokenizedDocumentInstance]:
        # tokenizer may return a list of tokenized instances in case of question answering
        if self.overflow_strategy == "return_first":
            return [tokenized_instance.resolve_overflow(0)]
        elif self.overflow_strategy == "return_random":
            import random

            random_idx = random.randint(0, tokenized_instance.batch_size)
            return [tokenized_instance.resolve_overflow(random_idx)]
        elif self.overflow_strategy == "return_all":
            return [
                tokenized_instance.resolve_overflow(i)
                for i in range(tokenized_instance.batch_size)
            ]
        else:
            raise ValueError(f"Unknown overflow strategy: {self.overflow_strategy}")


@DATA_TRANSFORMS.register("document_processor/sequence_classification")
class SequenceClassificationDocumentProcessor(DocumentProcessor):
    hf_processor: HuggingfaceProcessor = HuggingfaceProcessor()

    # segment-level-rank info args
    use_segment_level_bboxes: bool = False
    resize_image: tuple[int, int] | None = None
    load_image: bool = True
    load_bboxes: bool = True

    def _initialize_document_tokenizer(self) -> DocumentTokenizer:
        return SequenceClassificationDocumentTokenizer(
            hf_processor=self.hf_processor,
            use_segment_level_bboxes=self.use_segment_level_bboxes,
            resize_image=self.resize_image,
            load_image=self.load_image,
            load_bboxes=self.load_bboxes,
        )


@DATA_TRANSFORMS.register("document_processor/token_classification")
class TokenClassificationDocumentProcessor(DocumentProcessor):
    hf_processor: HuggingfaceProcessor = HuggingfaceProcessor()

    # segment-level-rank info args
    use_segment_level_bboxes: bool = False
    resize_image: tuple[int, int] | None = None
    load_image: bool = True
    load_bboxes: bool = True

    def _initialize_document_tokenizer(self) -> DocumentTokenizer:
        return TokenClassificationDocumentTokenizer(
            hf_processor=self.hf_processor,
            use_segment_level_bboxes=self.use_segment_level_bboxes,
            resize_image=self.resize_image,
            load_image=self.load_image,
            load_bboxes=self.load_bboxes,
        )


@DATA_TRANSFORMS.register("document_processor/question_answering")
class QuestionAnsweringDocumentProcessor(DocumentProcessor):
    hf_processor: HuggingfaceProcessor = HuggingfaceProcessor()

    # segment-level-rank info args
    use_segment_level_bboxes: bool = False
    resize_image: tuple[int, int] | None = None
    load_image: bool = True
    load_bboxes: bool = True
    ignore_no_answer_qa_pair: bool = False

    def _initialize_document_tokenizer(self) -> DocumentTokenizer:
        return QuestionAnsweringDocumentTokenizer(
            hf_processor=self.hf_processor,
            use_segment_level_bboxes=self.use_segment_level_bboxes,
            resize_image=self.resize_image,
            load_image=self.load_image,
            load_bboxes=self.load_bboxes,
            ignore_no_answer_qa_pair=self.ignore_no_answer_qa_pair,
        )
