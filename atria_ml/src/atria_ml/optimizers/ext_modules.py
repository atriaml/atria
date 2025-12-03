from atria_ml.registry import OPTIMIZER

for optimizer_name, optimizer_path in [
    ("adadelta", "torch.optim.Adadelta"),
    ("adam", "torch.optim.Adam"),
    ("adamw", "torch.optim.AdamW"),
    ("sparse_adam", "torch.optim.SparseAdam"),
    ("adagrad", "torch.optim.Adagrad"),
    ("adamax", "torch.optim.Adamax"),
    ("asgd", "torch.optim.ASGD"),
    ("lbfgs", "torch.optim.LBFGS"),
    ("rmsprop", "torch.optim.RMSprop"),
    ("rprop", "torch.optim.Rprop"),
    ("sgd", "torch.optim.SGD"),
    ("lars", "atria_ml.optimizers.lars.LARS"),
]:
    OPTIMIZER.register(name=optimizer_name)(optimizer_path)
