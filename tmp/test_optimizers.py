from atria_ml.optimizers._api import load_optimizer_config
from torch import nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)


model = Model()

opt = load_optimizer_config("sgd")
opt.build(model.parameters(), lr=0.1, momentum=0.9)
print(opt)
