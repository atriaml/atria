from typing import Optional

import torch
from pydantic import model_validator

from atria_transforms.types._base import MetadataBase, TensorDataModel


class DocumentTensorDataModel(TensorDataModel):
    class Metadata(MetadataBase):
        sample_id: str
        words: list[str] = None

    token_ids: torch.Tensor
    word_ids: torch.Tensor
    sequence_ids: torch.Tensor
    token_bboxes: torch.Tensor | None = None
    token_type_ids: torch.Tensor | None = None
    token_labels: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    overflow_to_sample_mapping: torch.Tensor | None = None

    # segment level fields
    segment_index: torch.Tensor | None = None
    segment_inner_token_rank: torch.Tensor | None = None
    first_token_idxes: torch.Tensor | None = None
    first_token_idxes_mask: torch.Tensor | None = None

    # sample level fields
    image: torch.Tensor
    label: torch.Tensor | None = None

    # extractive QA specific fields
    question_id: int | None = None
    qa_question: str | None = None
    qa_answers: list[str] | None = None
    token_answer_start: Optional["torch.Tensor"] = None
    token_answer_end: Optional["torch.Tensor"] = None

    @model_validator(mode="after")
    def validate_tensor_fields(self) -> "DocumentTensorDataModel":
        # ensure that each of the tensor fields are tensors of batch size 1
        for name, _ in self.__class__.model_fields.items():
            if name == "metadata":
                continue
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"Field '{name}' must be torch.Tensor, got {type(value).__name__}"
                    )
                if value.shape[0] != 1:
                    raise ValueError(
                        f"Field '{name}' must have batch size 1, got {value.shape[0]}"
                    )
        return self
