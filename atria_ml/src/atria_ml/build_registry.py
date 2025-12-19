from atria_ml.optimizers._configs import *  # noqa: F401
from atria_ml.optimizers._registry_group import OPTIMIZERS
from atria_ml.schedulers._configs import *  # noqa: F401
from atria_ml.schedulers._registry_group import LR_SCHEDULERS

if __name__ == "__main__":
    OPTIMIZERS.dump()
    LR_SCHEDULERS.dump()
