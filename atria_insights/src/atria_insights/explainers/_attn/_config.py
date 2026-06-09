from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, TypeVar

from atria_registry import ModuleConfig
from pydantic import Field

from atria_insights.explainers._base import ExplainerConfig
from atria_insights.explainers._registry_group import EXPLAINERS


class AttnExplainerConfig(ExplainerConfig):
    __builds_with_kwargs__: bool = True
    head_reduction: Literal["mean", "max", "min", "sum"] = "mean"

    @property
    def name(self) -> str:
        return f"{self.type.replace('/', '.')}.hr_{self.head_reduction}"

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"module_path", "type"})

    def build(  # type: ignore[override]
        self, model, **kwargs
    ):
        return ModuleConfig.build(self, model=model, **kwargs)


T_AttnExplainerConfig = TypeVar("T_AttnExplainerConfig", bound=AttnExplainerConfig)


@EXPLAINERS.register("attn/raw_attention")
class AttentionExplainerConfig(AttnExplainerConfig):
    __module_path__: ClassVar[str] = (
        "atria_insights.explainers._attn._base_explainer.AttentionExplainer"
    )
    type: Literal["attn/raw_attention"] = "attn/raw_attention"


@EXPLAINERS.register("attn/attention_rollout")
class AttentionRolloutExplainerConfig(AttnExplainerConfig):
    __module_path__: ClassVar[str] = (
        "atria_insights.explainers._attn._base_explainer.AttentionRolloutExplainer"
    )
    type: Literal["attn/attention_rollout"] = "attn/attention_rollout"


@EXPLAINERS.register("attn/attention_flow")
class AttentionFlowExplainerConfig(AttnExplainerConfig):
    __module_path__: ClassVar[str] = (
        "atria_insights.explainers._attn._base_explainer.AttentionFlowExplainer"
    )
    type: Literal["attn/attention_flow"] = "attn/attention_flow"


AttnExplainerConfigType = Annotated[
    AttentionExplainerConfig
    | AttentionRolloutExplainerConfig
    | AttentionFlowExplainerConfig,
    Field(discriminator="type"),
]
