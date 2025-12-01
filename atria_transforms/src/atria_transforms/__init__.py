# # ruff: noqa

# from typing import TYPE_CHECKING

# import lazy_loader as lazy

# from .registry import DATA_TRANSFORM

# if TYPE_CHECKING:
#     from .core.image import (
#         Cifar10ImageTransform,
#         FixedAspectRatioResize,
#         TensorGrayToRgb,
#     )
#     from .core.mmdet import (
#         DocumentInstanceMMDetTransform,
#         MMDetInput,
#         RandomChoiceResize,
#     )
#     from .core.sequence import DocumentInstanceTokenizer, TokenizerObjectSanitizer
#     from .core.torchvision import TorchvisionTransform
#     from .types import TokenizedDocumentInstance, TokenizedQuestionAnswerPair

# __getattr__, __dir__, __all__ = lazy.attach(
#     __name__,
#     submod_attrs={
#         "core.image": [
#             "Cifar10ImageTransform",
#             "FixedAspectRatioResize",
#             "TensorGrayToRgb",
#         ],
#         "core.mmdet": [
#             "DocumentInstanceMMDetTransform",
#             "MMDetInput",
#             "RandomChoiceResize",
#         ],
#         "core.sequence": ["DocumentInstanceTokenizer", "TokenizerObjectSanitizer"],
#         "core.torchvision": ["TorchvisionTransform"],
#         "data_types": ["TokenizedDocumentInstance", "TokenizedQuestionAnswerPair"],
#     },
# )
