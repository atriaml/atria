from typing import TYPE_CHECKING

import lazy_loader as lazy

if TYPE_CHECKING:
    from atria_transforms.tfs._hf_processor import HuggingfaceProcessor  # noqa
    from atria_transforms.tfs._image_processor import ImageProcessor  # noqa
    from atria_transforms.tfs._document_processor._base import DocumentProcessor  # noqa
    from atria_transforms.tfs._document_processor._task_tfs import (
        SequenceClassificationDocumentProcessor,
        TokenClassificationDocumentProcessor,
        QuestionAnsweringDocumentProcessor,
    )
    from ._torchvision import ResizeTransform  # noqa


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_hf_processor": ["HuggingfaceProcessor"],
        "_image_processor": ["ImageProcessor"],
        "_document_processor._base": ["DocumentProcessor"],
        "_document_processor._task_tfs": [
            "SequenceClassificationDocumentProcessor",
            "TokenClassificationDocumentProcessor",
            "QuestionAnsweringDocumentProcessor",
        ],
        "_torchvision": ["ResizeTransform"],
    },
)
