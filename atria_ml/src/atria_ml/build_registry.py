from atria_ml.optimizers._torch import *  # noqa: F401
from atria_ml.registry import LR_SCHEDULER, OPTIMIZER
from atria_ml.schedulers._torch import *  # noqa: F401

if __name__ == "__main__":
    OPTIMIZER.dump()
    LR_SCHEDULER.dump()
