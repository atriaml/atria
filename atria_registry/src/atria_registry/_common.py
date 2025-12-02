from typing import TypeVar

from ._module_base import RegisterableModule, RegisterablePydanticModule
from ._module_builder import ModuleBuilder

T_RegisterableModule = TypeVar(
    "T_RegisterableModule",
    bound=RegisterableModule | RegisterablePydanticModule | ModuleBuilder,
)
