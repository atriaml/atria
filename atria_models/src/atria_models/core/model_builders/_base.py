"""Atria Model Builder Base Class Module"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atria_logger import get_logger
from rich.pretty import pretty_repr

from atria_models.core.model_builders._common import FrozenLayers, ModelBuilderType
from atria_models.core.model_builders._constants import _DEFAULT_ATRIA_MODELS_CACHE_DIR
from atria_models.utilities._checkpoints import _load_checkpoint_from_path_or_url
from atria_models.utilities._common import _resolve_module_from_path

if TYPE_CHECKING:
    from torch import nn


logger = get_logger(__name__)


class ModelBuilder:
    def __init__(
        self,
        cache_dir: str | None = None,
        bn_to_gn: bool = False,
        frozen_layers: FrozenLayers | list[str] = FrozenLayers.none,
        pretrained_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        self._cache_dir = cache_dir or _DEFAULT_ATRIA_MODELS_CACHE_DIR
        self._bn_to_gn = bn_to_gn
        self._frozen_layers = frozen_layers
        self._pretrained_checkpoint = pretrained_checkpoint

    @classmethod
    def from_type(cls, builder_type: ModelBuilderType, **kwargs: Any):
        if builder_type == ModelBuilderType.local:
            return cls(**kwargs)
        elif builder_type == ModelBuilderType.timm:
            from atria_models.core.model_builders._timm import TimmModelBuilder

            return TimmModelBuilder(**kwargs)
        elif builder_type == ModelBuilderType.torchvision:
            from atria_models.core.model_builders._torchvision import (
                TorchvisionModelBuilder,
            )

            return TorchvisionModelBuilder(**kwargs)
        elif builder_type == ModelBuilderType.transformers_sequence:
            from atria_models.core.model_builders._transformers import (
                SequenceClassificationModelBuilder,
            )

            return SequenceClassificationModelBuilder(**kwargs)
        elif builder_type == ModelBuilderType.transformers_token_classification:
            from atria_models.core.model_builders._transformers import (
                TokenClassificationModelBuilder,
            )

            return TokenClassificationModelBuilder(**kwargs)
        elif builder_type == ModelBuilderType.transformers_question_answering:
            from atria_models.core.model_builders._transformers import (
                QuestionAnsweringModelBuilder,
            )

            return QuestionAnsweringModelBuilder(**kwargs)
        elif builder_type == ModelBuilderType.transformers_image_classification:
            from atria_models.core.model_builders._transformers import (
                ImageClassificationModelBuilder,
            )

            return ImageClassificationModelBuilder(**kwargs)
        else:
            raise ValueError(f"Unsupported ModelBuilderType: {builder_type}")

    def _validate_model(self, model: Any) -> nn.Module:
        from torch.nn import Module

        if not isinstance(model, Module):
            raise ValueError(f"Model is not a valid PyTorch module. Got {type(model)}")
        return model

    def _configure_batch_norm_layers(self, model: nn.Module) -> None:
        from atria_models.utilities._nn_modules import _batch_norm_to_group_norm

        if self._bn_to_gn:
            logger.warning(
                "Converting BatchNorm layers to GroupNorm layers in the model. "
                "If this is not intended, set convert_bn_to_gn=False."
            )
            _batch_norm_to_group_norm(model)

    def _configure_model_frozen_layers(self, model: nn.Module) -> None:
        import json

        from atria_models.utilities._nn_modules import _freeze_layers_with_key_pattern

        if self._frozen_layers == FrozenLayers.all:
            logger.warning(
                "Freezing the model. If this is not intended, set is_frozen=False in its config."
            )
            model.requires_grad_(False)
        elif isinstance(self._frozen_layers, list):
            if self._frozen_layers:
                trainable_params = _freeze_layers_with_key_pattern(
                    model=model, frozen_layers=self._frozen_layers
                )
                logger.info(
                    f"Trainable parameters: {json.dumps(trainable_params, indent=2)}"
                )

    def build(self, model_name_or_path: str, **kwargs) -> nn.Module:
        import ignite.distributed as idist

        if idist.get_rank() > 0:
            idist.barrier()

        model = self._build(model_name_or_path=model_name_or_path, **kwargs)

        if idist.get_rank() == 0:
            idist.barrier()

        # Validate the model type.
        model = self._validate_model(model)

        # Load pretrained checkpoint if specified.
        if self._pretrained_checkpoint is not None:
            checkpoint = _load_checkpoint_from_path_or_url(self._pretrained_checkpoint)
            missing_keys, unexpected_keys = model.load_state_dict(
                checkpoint, strict=False
            )
            if missing_keys or unexpected_keys:
                logger.warning(
                    "Model loaded with missing or unexpected keys:\n"
                    f"Missing keys: {missing_keys}\n"
                    f"Unexpected keys: {unexpected_keys}"
                )

        # Configure BatchNorm layers if specified.
        self._configure_batch_norm_layers(model)

        # Configure frozen layers if specified.
        self._configure_model_frozen_layers(model)
        return model

    def _build(self, model_name_or_path: str, **kwargs) -> nn.Module:
        import inspect

        module: type = _resolve_module_from_path(model_name_or_path)
        signature = inspect.signature(module.__init__)
        valid_params = signature.parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        logger.info(
            f"Building model '{model_name_or_path}' with parameters:\n{pretty_repr(filtered_kwargs, expand_all=True)}"
        )
        return module(**filtered_kwargs)
