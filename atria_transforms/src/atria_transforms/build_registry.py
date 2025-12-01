from pathlib import Path

from atria_registry.utilities import write_registry_to_yaml

from atria_transforms.core.image import *  # noqa
from atria_transforms.core.mmdet import *  # noqa
from atria_transforms.core.sequence import *  # noqa
from atria_transforms.core.torchvision import *  # noqa
from atria_transforms.registry import DATA_TRANSFORM

if __name__ == "__main__":
    write_registry_to_yaml(Path(__file__).parent / "conf", types=[DATA_TRANSFORM.name])
