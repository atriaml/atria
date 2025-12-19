from typing import TypeVar

from ._module_base import ModuleConfig, PydanticConfigurableModule

T_ModuleConfig = TypeVar(
    "T_ModuleConfig", bound=ModuleConfig | PydanticConfigurableModule
)
