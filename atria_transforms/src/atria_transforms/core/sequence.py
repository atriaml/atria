from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atria_core.logger.logger import get_logger
from atria_core.transforms import DataTransform
from atria_core.types import TaskType
from atria_core.types.generic.bounding_box import BoundingBoxList
from atria_core.types.generic.ground_truth import GroundTruth
from atria_registry import RegistryConfig
from pydantic import Field

from atria_transforms.registry import DATA_TRANSFORM

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    import torch
    from atria_core.types import DocumentInstance, TaskType

    from atria_transforms.data_types import (
        TokenizedDocumentInstance,
        TokenizedQuestionAnswerPair,
    )

logger = get_logger(__name__)


@DATA_TRANSFORM.register("tokenizer_object_sanitizer")
class TokenizerObjectSanitizer(DataTransform):
    """
    This class is used to sanitize the tokenized sequences to match the original sequence elements.
    For example, if you have labels for words, you can map them to the tokens.
    [0, 1, 2] -> ['hello', 'world', '!']
    This is mapped to tokens like so
    [<padding_element>, 0, <padding_element>, 1,  <padding_element>, 2, <padding_element>, ...] ->
    [<cls-token>, '#hel', '#lo', '#wor', '#ld' '!', <pad-token>, ...]
    """

    apply_value_to_first_subword: bool = True

    def _sanitize_sequence(
        self, sequence: torch.Tensor, word_ids: torch.Tensor, padding_value: Any = -100
    ):
        import torch

        sanitized = torch.ones(sequence.shape) * padding_value
        last_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id == -100:
                continue
            if (
                last_word_id is not None
                and last_word_id == word_id
                and self.apply_value_to_first_subword
            ):
                continue
            sanitized[idx] = sequence[word_id]
            last_word_id = word_id
        return sanitized

    def _apply_transform(
        self, tokenized_object_instance: TokenizedDocumentInstance
    ) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        tokenized_object_instance.token_ids = self._sanitize_sequence(
            tokenized_object_instance.token_ids,
            tokenized_object_instance.word_ids,
            padding_value=-100,
        )
        tokenized_object_instance.token_bboxes = self._sanitize_sequence(
            tokenized_object_instance.token_bboxes,
            tokenized_object_instance.word_ids,
            padding_value=[-100, -100, -100, -100],
        )
        tokenized_object_instance.token_labels = self._sanitize_sequence(
            tokenized_object_instance.token_labels,
            tokenized_object_instance.word_ids,
            padding_value=-100,
        )
        return tokenized_object_instance


@DATA_TRANSFORM.register(
    "document_instance_tokenizer",
    configs=[
        RegistryConfig(
            name=TaskType.sequence_classification.value,
            task_type=TaskType.sequence_classification.value,
        ),
        RegistryConfig(
            name=TaskType.token_classification.value,
            task_type=TaskType.token_classification.value,
        ),
        RegistryConfig(
            name=TaskType.layout_token_classification.value,
            task_type=TaskType.layout_token_classification.value,
        ),
        RegistryConfig(
            name=TaskType.question_answering.value,
            task_type=TaskType.question_answering.value,
            call_kwargs={"stride": 128, "truncation": "only_second"},
        ),
        RegistryConfig(
            name=TaskType.visual_question_answering.value,
            task_type=TaskType.visual_question_answering.value,
            call_kwargs={"stride": 128, "truncation": "only_second"},
        ),
    ],
)
class DocumentInstanceTokenizer(DataTransform):
    tokenizer_name: str
    task_type: TaskType
    init_kwargs: dict = Field(default_factory=dict)
    call_kwargs: dict = Field(default_factory=dict)
    overflow_sampling: str = "return_all"
    max_overflow_samples: int = 10
    use_ssl: bool = False  # Use segment-level bounding boxes for SER and VQA
    do_normalize: bool = True  # Normalize the image to ImageNet mean and std
    do_resize: bool = True  # Resize the image to 224x224
    resize_height: int = 224
    resize_width: int = 224
    image_mean: list[float] | None = None
    image_std: list[float] | None = None

    def model_post_init(self, context) -> None:
        # lazy initialization of the processor to avoid significant transformers loading times
        import os

        from atria_core.constants import _DEFAULT_ATRIA_MODELS_CACHE_DIR
        from transformers.utils.constants import (
            IMAGENET_DEFAULT_MEAN,
            IMAGENET_DEFAULT_STD,
        )

        self.image_mean = self.image_mean or IMAGENET_DEFAULT_MEAN
        self.image_std = self.image_std or IMAGENET_DEFAULT_STD

        # how to return overflowing samples?
        assert self.overflow_sampling in [
            "return_all",
            "return_random_n",
            "no_overflow",
            "return_first_n",
        ], f"Overflow sampling strategy {self.overflow_sampling} is not supported."

        # initialize the tokenizer
        default_init_kwargs = {
            "cache_dir": os.path.join(_DEFAULT_ATRIA_MODELS_CACHE_DIR, "transformers"),
            "local_files_only": False,
            "apply_ocr": False,
            "image_mean": IMAGENET_DEFAULT_MEAN,
            "image_std": IMAGENET_DEFAULT_STD,
            "add_prefix_space": True,
            "do_lower_case": True,
            "do_normalize": False,  # normalize means imagenet normalization.
            "do_resize": False,  # resize means resizing to 224x224.
            "do_rescale": False,  # rescale means division by 1/255.
        }

        # setup the default call kwargs
        default_call_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": 512,
            "stride": 0,
            "pad_to_multiple_of": 8,
            "is_split_into_words": True,
            "return_overflowing_tokens": self.overflow_sampling
            != "no_overflow",  # set some arguments that we need to stay fixed for our case
            "return_token_type_ids": None,
            "return_attention_mask": None,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_length": False,
            "return_tensors": "pt",
            "verbose": True,
        }
        self.init_kwargs = {**default_init_kwargs, **self.init_kwargs}
        self.call_kwargs = {**default_call_kwargs, **self.call_kwargs}

    def _lazy_post_init(self) -> None:
        import inspect

        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.tokenizer_name, **self.init_kwargs
        )
        possible_args = inspect.signature(self._processor.__call__).parameters
        for key in list(self.call_kwargs.keys()):
            if key not in possible_args:
                logger.warning(
                    f"Invalid keyword argument '{key}' found in call_kwargs for {self.__class__.__name__}. Skipping it."
                )
                self.call_kwargs.pop(key)

    def _prepare_ocr_inputs(
        self, ground_truth: GroundTruth, task_type: TaskType
    ) -> tuple[list[str], BoundingBoxList]:
        if self.task_type == TaskType.sequence_classification:
            assert ground_truth.ocr is not None, (
                "Ground truth of type 'ocr' is required for the task type 'sequence_classification'."
                " Available ground truth types: {}"
            ).format(
                [
                    key
                    for key, value in ground_truth.__dict__.items()
                    if value is not None
                ]
            )
            return {
                "words": ground_truth.ocr.words,
                "word_bboxes": ground_truth.ocr.word_bboxes,
            }
        elif self.task_type in [
            TaskType.token_classification,
            TaskType.layout_token_classification,
        ]:
            assert ground_truth.ser is not None or ground_truth.ocr is not None, (
                "Ground truth of type 'ser' is required for the task type 'token_classification'."
                " Available ground truth types: {}"
            ).format(
                [
                    key
                    for key, value in ground_truth.__dict__.items()
                    if value is not None
                ]
            )
            if ground_truth.ser is not None:
                return {
                    "words": ground_truth.ser.words,
                    "word_bboxes": ground_truth.ser.segment_level_bboxes
                    if self.use_ssl
                    and ground_truth.ser.segment_level_bboxes is not None
                    else ground_truth.ser.word_bboxes,
                    "word_labels": ground_truth.ser.word_labels,
                }
            else:
                return {
                    "words": ground_truth.ocr.words,
                    "word_bboxes": ground_truth.ocr.word_bboxes,
                }
        elif self.task_type == TaskType.question_answering:
            assert ground_truth.qa is not None, (
                "Ground truth of type 'qa' is required for the task type 'visual_question_answering'."
                " Available ground truth types: {}"
            ).format(
                [
                    key
                    for key, value in ground_truth.__dict__.items()
                    if value is not None
                ]
            )
            return {
                "question_text": ground_truth.qa.qa_pair.question_text,
                "words": ground_truth.qa.words,
            }
        elif self.task_type == TaskType.visual_question_answering:
            assert ground_truth.vqa is not None or ground_truth.ocr is not None, (
                "Ground truth of type 'vqa' is required for the task type 'visual_question_answering'."
                " Available ground truth types: {}"
            ).format(
                [
                    key
                    for key, value in ground_truth.__dict__.items()
                    if value is not None
                ]
            )
            return {
                "question_text": ground_truth.vqa.qa_pair.question_text,
                "words": ground_truth.vqa.words,
                "word_bboxes": ground_truth.vqa.segment_level_bboxes
                if self.use_ssl and ground_truth.vqa.segment_level_bboxes is not None
                else ground_truth.vqa.word_bboxes,
            }
        else:
            raise ValueError(
                f"Task type {self.task_type} is not supported. "
                "Supported task types are: {}".format(
                    [
                        TaskType.sequence_classification,
                        TaskType.token_classification,
                        TaskType.question_answering,
                        TaskType.visual_question_answering,
                    ]
                )
            )

    def _prepare_tokenizer_inputs(
        self, document_instance: DocumentInstance
    ) -> Mapping[str, Any]:
        from atria_core.types import TaskType

        ground_truth = document_instance.gt
        document_instance.image.load().to_rgb().to_tensor()
        if self.do_resize:
            document_instance.image.resize(
                width=self.resize_width, height=self.resize_height
            )

        if self.do_normalize:
            document_instance.image.normalize(mean=self.image_mean, std=self.image_std)

        ocr_inputs = self._prepare_ocr_inputs(
            ground_truth=ground_truth, task_type=self.task_type
        )

        tokenizer_inputs = {"images": document_instance.image.content}
        if self.task_type == TaskType.sequence_classification:
            tokenizer_inputs["text"] = ocr_inputs["words"]
            tokenizer_inputs["boxes"] = ocr_inputs["word_bboxes"].value
        elif self.task_type in [
            TaskType.token_classification,
            TaskType.layout_token_classification,
        ]:
            tokenizer_inputs["text"] = ocr_inputs["words"]
            tokenizer_inputs["boxes"] = ocr_inputs["word_bboxes"].value
            tokenizer_inputs["word_labels"] = ocr_inputs.get("word_labels", None).value
        elif self.task_type == TaskType.question_answering:
            tokenizer_inputs["text"] = ocr_inputs["question_text"]
            tokenizer_inputs["text_pair"] = ocr_inputs["words"]
        elif self.task_type == TaskType.visual_question_answering:
            tokenizer_inputs["text"] = ocr_inputs["question_text"]
            tokenizer_inputs["text_pair"] = ocr_inputs["words"]
            tokenizer_inputs["boxes"] = ocr_inputs["word_bboxes"].value
        else:
            raise ValueError(
                f"Task type {self.task_type} is not supported. "
                "Supported task types are: {}".format(
                    [
                        TaskType.sequence_classification,
                        TaskType.token_classification,
                        TaskType.question_answering,
                        TaskType.visual_question_answering,
                    ]
                )
            )
        return tokenizer_inputs

    def _get_subword_start_end(self, word_start, word_end, word_ids, sequence_ids):
        start_of_context = -1
        for i in range(len(sequence_ids)):
            if sequence_ids[i] == 1:
                start_of_context = i
                break
        num_question_tokens = start_of_context
        assert start_of_context != -1, "Could not find the start of the context"
        subword_start = -1
        subword_end = -1
        for i in range(start_of_context, len(word_ids)):
            if word_start == word_ids[i] and subword_start == -1:
                subword_start = i
            if word_end == word_ids[i]:
                subword_end = i
        return subword_start, subword_end, num_question_tokens

    def _generate_qa_token_ids(
        self,
        qa_pair: TokenizedQuestionAnswerPair,
        word_ids: torch.Tensor,
        sequence_ids: torch.Tensor,
        sequence_length: int = 512,
    ) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        import torch

        tokenized_answer_starts, tokenized_answer_ends = [], []
        for word_ids_per_overflow, sequence_ids_per_overflow in zip(
            word_ids, sequence_ids, strict=True
        ):
            tokenized_answer_start, tokenized_answer_end = None, None
            if qa_pair.answer_start == -1:
                tokenized_answer_start = 0
                tokenized_answer_end = 0
            else:
                (tokenized_answer_start, tokenized_answer_end, _) = (
                    self._get_subword_start_end(
                        qa_pair.answer_start,
                        qa_pair.answer_end,
                        word_ids_per_overflow,
                        sequence_ids_per_overflow,
                    )
                )
                if tokenized_answer_start == -1:
                    tokenized_answer_start = 0
                    tokenized_answer_end = 0
                if tokenized_answer_end == -1:
                    tokenized_answer_end = sequence_length - 1
                assert tokenized_answer_end >= tokenized_answer_start, (
                    "End token index is less than start token index. "
                    "Something is wrong in the conversion from answer word indices to answer token indices."
                )
            tokenized_answer_starts.append(tokenized_answer_start)
            tokenized_answer_ends.append(tokenized_answer_end)
        tokenized_answer_start = torch.tensor(
            tokenized_answer_starts, dtype=torch.long, device=word_ids.device
        )
        tokenized_answer_end = torch.tensor(
            tokenized_answer_ends, dtype=torch.long, device=word_ids.device
        )
        return tokenized_answer_start, tokenized_answer_end

    def _apply_transforms(
        self, document_instance: DocumentInstance
    ) -> Mapping[str, Any] | list[Mapping[str, Any]]:
        import torch
        from atria_core.types import TaskType

        from atria_transforms.data_types import TokenizedDocumentInstance

        assert not document_instance._is_batched, (
            f"Document instance {document_instance.id} is batched. "
            "This tokenizer only supports single document instances."
        )

        tokenized_samples_batch = self._processor(
            **self._prepare_tokenizer_inputs(document_instance), **self.call_kwargs
        )
        tokenized_samples_batch["word_ids"] = torch.tensor(
            [
                [-100 if x is None else x for x in tokenized_samples_batch.word_ids(i)]
                for i in range(len(tokenized_samples_batch["input_ids"]))
            ]
        )
        tokenized_samples_batch["sequence_ids"] = torch.tensor(
            [
                [
                    -100 if x is None else x
                    for x in tokenized_samples_batch.sequence_ids(i)
                ]
                for i in range(len(tokenized_samples_batch["input_ids"]))
            ]
        )

        # rename some fields
        tokenized_samples_batch["token_ids"] = tokenized_samples_batch.pop("input_ids")
        tokenized_samples_batch["token_bboxes"] = tokenized_samples_batch.pop(
            "bbox", None
        )
        tokenized_samples_batch["token_type_ids"] = tokenized_samples_batch.pop(
            "token_type_ids", None
        )
        tokenized_samples_batch["token_labels"] = tokenized_samples_batch.pop(
            "labels", None
        )
        if "pixel_values" in tokenized_samples_batch:
            document_instance.image.content = tokenized_samples_batch.pop(
                "pixel_values"
            )[0]

        additional_kwargs = {}
        if self.task_type == TaskType.visual_question_answering:
            qa_pair = document_instance.gt.vqa.qa_pair
            tokenized_answer_start, tokenized_answer_end = self._generate_qa_token_ids(
                qa_pair=qa_pair,
                word_ids=tokenized_samples_batch["word_ids"],
                sequence_ids=tokenized_samples_batch["sequence_ids"],
                sequence_length=tokenized_samples_batch["token_ids"].shape[-1],
            )
            additional_kwargs = {
                "qa_pair": qa_pair,
                "tokenized_answer_start": tokenized_answer_start,
                "tokenized_answer_end": tokenized_answer_end,
            }

        ocr_inputs = self._prepare_ocr_inputs(
            ground_truth=document_instance.gt, task_type=self.task_type
        )
        document_instance = TokenizedDocumentInstance(
            # copy the document instance attributes
            index=document_instance.index,
            sample_id=document_instance.sample_id,
            image=document_instance.image,
            words=ocr_inputs.get("words", None),
            word_bboxes=ocr_inputs.get("word_bboxes", None),
            label=(
                document_instance.gt.classification.label
                if document_instance.gt.classification is not None
                else None
            ),
            # copy the tokenized samples batch attributes
            **tokenized_samples_batch,
            **additional_kwargs,
        )
        document_instance._tokenizer = self._processor.tokenizer
        document_instance._tokenizer_config = {
            k: v
            for k, v in self.call_kwargs.items()
            if k
            in [
                "add_special_tokens",
                "max_length",
                "padding",
                "truncation",
                "stride",
                "is_split_into_words",
            ]
        }

        return document_instance

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(tokenizer_name={self.tokenizer_name}, "
            f"task_type={self.task_type}, init_kwargs={self.init_kwargs}, "
            f"call_kwargs={self.call_kwargs}, overflow_sampling={self.overflow_sampling}, "
            f"max_overflow_samples={self.max_overflow_samples}, "
        )
