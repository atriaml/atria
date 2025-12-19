# noqa: D104
from typing import TYPE_CHECKING

import lazy_loader as lazy

from atria_transforms import registry

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
    from atria_transforms.tfs._torchvision import ResizeTransform  # noqa
    from atria_transforms.api.tfs import load_transform

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "api.tfs": ["load_transform"],
        "tfs._hf_processor": ["HuggingfaceProcessor"],
        "tfs._image_transforms": ["StandardImageTransform"],
        "tfs._document_processor._base": ["DocumentProcessor"],
        "tfs._document_processor._task_tfs": [
            "SequenceClassificationDocumentProcessor",
            "TokenClassificationDocumentProcessor",
            "QuestionAnsweringDocumentProcessor",
        ],
        "tfs._image_processor._base": ["ImageProcessor"],
        "tfs._torchvision": ["ResizeTransform"],
    },
)
