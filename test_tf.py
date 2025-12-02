# # # from atria_transforms.tfs.document_processor._task_tfs import *  # noqa

# import torch

# from atria_transforms import load_transform

# tf = load_transform("resize/default", size=(128, 128))
# print("tf", tf)
# x = torch.randn(3, 224, 224)
# print("x.shape", x.shape)
# x = tf(x)
# print("x.shape", x.shape)

# from atria_transforms import DocumentProcessor, HuggingfaceProcessor, ImageProcessor

# tf = ImageProcessor()
# tf2 = HuggingfaceProcessor()
# print("tf", tf)
# print("tf2", tf2)
# tf3 = DocumentProcessor()

# # from atria_transforms.tfs.document_processor._task_tfs import (
# #     SequenceClassificationDocumentProcessor,
# # )

# # tf = SequenceClassificationDocumentProcessor()


# from atria_transforms import *  # noqa
# # from torchvision.transforms import (
# #     CenterCrop,
# #     Normalize,
# #     RandomHorizontalFlip,
# #     Resize,
# #     ToTensor,
# # )

# # if TYPE_CHECKING:
# #     pass

# # for tf_name, tf in [
# #     ("CenterCrop", CenterCrop),
# #     ("Normalize", Normalize),
# #     ("RandomHorizontalFlip", RandomHorizontalFlip),
# #     ("Resize", Resize),
# #     ("ToTensor", ToTensor),
# # ]:
# #     DATA_TRANSFORM.register(tf_name)(tf)
