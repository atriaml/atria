import enum


class BaselineStrategy(str, enum.Enum):
    zeros = "zeros"
    ones = "ones"
    batch_mean = "batch_mean"
    random = "random"
    fixed = "fixed"
