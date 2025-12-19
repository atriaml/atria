# noqa: D104
from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from atria_transforms.tfs._hf_processor import HuggingfaceProcessor  # noqa
    from atria_transforms.tfs._image_transforms import StandardImageTransform  # noqa
    from atria_transforms.tfs._document_processor._base import DocumentProcessor  # noqa
    from atria_transforms.tfs._document_processor._task_tfs import (
        SequenceClassificationDocumentProcessor,
        TokenClassificationDocumentProcessor,
        QuestionAnsweringDocumentProcessor,
    )
    from atria_transforms.tfs._image_transforms import StandardImageTransform  # noqa
    from ._torchvision import ResizeTransform  # noqa


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_hf_processor": ["HuggingfaceProcessor"],
        "_image_transforms": ["StandardImageTransform"],
        "_document_processor._base": ["DocumentProcessor"],
        "_document_processor._task_tfs": [
            "SequenceClassificationDocumentProcessor",
            "TokenClassificationDocumentProcessor",
            "QuestionAnsweringDocumentProcessor",
        ],
        "_image_processor._base": ["ImageProcessor"],
        "_torchvision": ["ResizeTransform"],
    },
)
