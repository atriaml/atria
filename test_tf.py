# # from atria_transforms.tfs.document_processor._task_tfs import *  # noqa

# # from atria_transforms import load_transform

# # tf = load_transform("document_processor/sequence_classification")
# # print("tf", tf)

from atria_transforms.core._tfs._hf_processor import HuggingfaceProcessor
from atria_transforms.core._tfs._image_processor import ImageProcessor
from atria_transforms.tfs.document_processor._base import DocumentProcessor

tf = ImageProcessor()
tf2 = HuggingfaceProcessor()
print("tf", tf)
print("tf2", tf2)
tf3 = DocumentProcessor()

from atria_transforms.tfs.document_processor._task_tfs import (
    SequenceClassificationDocumentProcessor,
)

tf = SequenceClassificationDocumentProcessor()
