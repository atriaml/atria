from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    import torch
    from torch import nn

logger = get_logger(__name__)


class ModelDict(BaseModel):
    """
    A data model for managing trainable and non-trainable PyTorch models.

    Attributes:
        trainable_models (nn.ModuleDict): A dictionary of trainable models.
        non_trainable_models (nn.ModuleDict): A dictionary of non-trainable models.
    """

    trainable_models: nn.ModuleDict
    non_trainable_models: nn.ModuleDict

    @field_validator("trainable_models", "non_trainable_models")
    @classmethod
    def check_module_dict(cls, v: nn.ModuleDict) -> nn.ModuleDict:
        """
        Validates that the provided value is an nn.ModuleDict.

        Args:
            v (nn.ModuleDict): The value to validate.

        Returns:
            nn.ModuleDict: The validated value.

        Raises:
            ValueError: If the value is not an nn.ModuleDict.
        """
        from torch import nn

        if not isinstance(v, nn.ModuleDict):
            raise ValueError("Must be a nn.ModuleDict")
        return v


def _rsetattr(obj: Any, attr: str, val: Any) -> None:
    """
    Recursively sets an attribute on an object.

    Args:
        obj (Any): The object to set the attribute on.
        attr (str): The attribute name, which can include nested attributes separated by dots.
        val (Any): The value to set for the attribute.
    """
    pre, _, post = attr.rpartition(".")
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def _rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    """
    Recursively gets an attribute from an object.

    Args:
        obj (Any): The object to get the attribute from.
        attr (str): The attribute name, which can include nested attributes separated by dots.
        *args (Any): Default value to return if the attribute is not found.

    Returns:
        Any: The value of the attribute.
    """

    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def _set_module_with_name(
    model: nn.Module, module_name: str, module: nn.Module
) -> None:
    """
    Sets a module in a model by name.

    Args:
        model (nn.Module): The model to modify.
        module_name (str): The name of the module to set.
        module (nn.Module): The module to set.
    """
    return setattr(model, module_name, module)


def _replace_module_with_name(
    module: nn.Module, target_name: str, new_module: nn.Module
) -> None:
    """
    Replaces a module in a model by name.

    Args:
        module (nn.Module): The parent module containing the target module.
        target_name (str): The name of the module to replace, which can include nested names separated by dots.
        new_module (nn.Module): The new module to replace the target module with.
    """
    target_name_split = target_name.split(".")
    if len(target_name_split) > 1:
        _replace_module_with_name(
            getattr(module, target_name_split[0]),
            ".".join(target_name_split[1:]),
            new_module,
        )
    setattr(module, target_name_split[-1], new_module)


def _get_last_module(model: nn.Module) -> Any:
    """
    Retrieves the last module in a model.

    Args:
        model (nn.Module): The model to search.

    Returns:
        Any: The last module in the model.
    """
    return list(model.named_modules())[-1]


def _find_layer_in_model(model: nn.Module, layer_name: str) -> str:
    """
    Finds a specific layer in a model by name.

    Args:
        model (nn.Module): The model to search.
        layer_name (str): The name of the layer to find.

    Returns:
        str: The name of the layer.

    Raises:
        ValueError: If the layer is not found in the model.
    """
    layer = [x for x, m in model.named_modules() if x == layer_name]
    if len(layer) == 0:
        raise ValueError(f"Encoder layer {layer_name} not found in the model.")
    return layer[0]


def _freeze_layers(layers: list[nn.Module]) -> None:
    """
    Freezes a list of layers in a model.

    Args:
        layers (List[nn.Module]): A list of layers to freeze.
    """
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = False


def _batch_norm_to_group_norm(model: nn.Module) -> None:
    """
    Converts batch normalization layers to group normalization.

    Args:
        model (nn.Module): The model to modify.
    """
    import torch

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            num_channels = module.num_features

            def get_groups(num_channels: int, groups: int) -> int:
                """
                Recursively determines the number of groups for group normalization.

                Args:
                    num_channels (int): The number of channels.
                    groups (int): The initial number of groups.

                Returns:
                    int: The adjusted number of groups.
                """
                if num_channels % groups != 0:
                    groups = groups // 2
                    groups = get_groups(num_channels, groups)
                return groups

            groups = get_groups(num_channels, 32)
            gn = torch.nn.GroupNorm(groups, num_channels)
            _rsetattr(model, name, gn)


def _freeze_layers_with_key_pattern(
    model: nn.Module, frozen_layers: list[str]
) -> dict[str, bool]:
    """
    Apply layer-specific freezing/unfreezing based on name patterns.

    Args:
        model (nn.Module): The model to modify.
        frozen_keys_patterns (List[str]): Patterns for layers to freeze.
        unfrozen_keys_patterns (List[str]): Patterns for layers to unfreeze.

    Returns:
        Dict[str, bool]: A dictionary of parameter names and their `requires_grad` status.
    """
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in frozen_layers):
            param.requires_grad = False
    trainable_params = {
        name: param.requires_grad for name, param in model.named_parameters()
    }
    return trainable_params


def _auto_model(
    device: torch.device, model: torch.nn.Module, sync_bn: bool = False, **kwargs: Any
) -> torch.nn.Module:
    """Helper method to adapt provided model for non-distributed and distributed configurations (supporting
    all available backends from :meth:`~ignite.distributed.utils.available_backends()`).

    Internally, we perform to following:

    - send model to current :meth:`~ignite.distributed.utils.device()` if model's parameters are not on the device.
    - wrap the model to `torch DistributedDataParallel`_ for native torch distributed if world size is larger than 1.
    - wrap the model to `torch DataParallel`_ if no distributed context found and more than one CUDA devices available.
    - broadcast the initial variable states from rank 0 to all other processes if Horovod distributed framework is used.

    Args:
        model: model to adapt.
        sync_bn: if True, applies `torch convert_sync_batchnorm`_ to the model for native torch
            distributed only. Default, False. Note, if using Nvidia/Apex, batchnorm conversion should be
            applied before calling ``amp.initialize``.
        kwargs: kwargs to model's wrapping class: `torch DistributedDataParallel`_ or `torch DataParallel`_
            if applicable. Please, make sure to use acceptable kwargs for given backend.

    Returns:
        torch.nn.Module

    Examples:
        .. code-block:: python

            import ignite.distribted as idist

            model = idist.auto_model(model)

        In addition with NVidia/Apex, it can be used in the following way:

        .. code-block:: python

            import ignite.distribted as idist

            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
            model = idist.auto_model(model)

    .. _torch DistributedDataParallel: https://pytorch.org/docs/stable/generated/torch.nn.parallel.
        DistributedDataParallel.html
    .. _torch DataParallel: https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html
    .. _torch convert_sync_batchnorm: https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#
        torch.nn.SyncBatchNorm.convert_sync_batchnorm

    .. versionchanged:: 0.4.2

        - Added Horovod distributed framework.
        - Added ``sync_bn`` argument.

    .. versionchanged:: 0.4.3
        Added kwargs to ``idist.auto_model``.
    """
    import torch
    import torch.nn as nn
    from ignite.distributed import utils as idist
    from ignite.distributed.comp_models import (
        horovod as idist_hvd,
        native as idist_native,
    )

    # Put model's parameters to device if its parameters are not on the device
    if not all([p.device == device for p in model.parameters()]):
        model.to(device)

    # distributed data parallel model
    if idist.get_world_size() > 1:
        bnd = idist.backend()
        if idist.has_native_dist_support and bnd in (
            idist_native.NCCL,
            idist_native.GLOO,
            idist_native.MPI,
        ):
            if sync_bn:
                logger.info("Convert batch norm to sync batch norm")
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            if torch.cuda.is_available():
                if "device_ids" in kwargs:
                    raise ValueError(
                        f"Argument kwargs should not contain 'device_ids', but got {kwargs}"
                    )

                lrank = idist.get_local_rank()
                logger.info(
                    f"Apply torch DistributedDataParallel on model, device id: {lrank}"
                )
                kwargs["device_ids"] = [lrank]
            else:
                logger.info("Apply torch DistributedDataParallel on model")

            model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
        elif idist.has_hvd_support and bnd == idist_hvd.HOROVOD:
            import horovod.torch as hvd

            logger.info(
                "Broadcast the initial variable states from rank 0 to all other processes"
            )
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # not distributed but multiple GPUs reachable so data parallel model
    elif torch.cuda.device_count() > 1 and "cuda" in idist.device().type:
        logger.info("Apply torch DataParallel on model")
        model = torch.nn.parallel.DataParallel(model, **kwargs)

    return model
