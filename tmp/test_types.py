import torch
from atria_transforms.data_types._document import DocumentTensorDataModel

model = DocumentTensorDataModel(
    sample_id="sample_001",
    words=["This", "is", "a", "test", "document", "."],
    token_ids=torch.tensor([[101, 2023, 2003, 1037, 3231, 6251, 1012, 102]]),
    word_ids=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]),
    sequence_ids=torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0]]),
    image=torch.tensor([[[[0.0]]]]),
)
print(model)
