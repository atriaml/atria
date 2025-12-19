from atria_ml.optimizers._configs import *  # noqa: F401
from atria_ml.optimizers._registry_group import OPTIMIZER
from atria_ml.schedulers._configs import *  # noqa: F401
from atria_ml.schedulers._registry_group import LR_SCHEDULER

if __name__ == "__main__":
    OPTIMIZER.dump()
    LR_SCHEDULER.dump()
