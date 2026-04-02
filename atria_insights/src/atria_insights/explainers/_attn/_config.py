from __future__ import annotations

from typing import Annotated, Literal, TypeVar

from atria_registry import ModuleConfig
from pydantic import Field
from traitlets import Any

from atria_insights.explainers._registry_group import EXPLAINERS


class AttnExplainerConfig(ModuleConfig):
    __builds_with_kwargs__: bool = True
    head_reduction: Literal["mean", "max", "min", "sum"] = "mean"

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude={"module_path", "type"})

    def build(  # type: ignore[override]
        self, model, **kwargs
    ):
        return ModuleConfig.build(self, model=model, **kwargs)


T_AttnExplainerConfig = TypeVar("T_AttnExplainerConfig", bound=AttnExplainerConfig)


@EXPLAINERS.register("attn/raw_attention")
class RawAttentionExplainerConfig(AttnExplainerConfig):
    type: Literal["attn/raw_attention"] = "attn/raw_attention"
    module_path: str | None = (
        "atria_insights.explainers._attn._base_explainer.RawAttentionExplainer"
    )


@EXPLAINERS.register("attn/attention_rollout")
class AttentionRolloutExplainerConfig(AttnExplainerConfig):
    type: Literal["attn/attention_rollout"] = "attn/attention_rollout"
    module_path: str | None = (
        "atria_insights.explainers._attn._base_explainer.AttentionRolloutExplainer"
    )


@EXPLAINERS.register("attn/attention_flow")
class AttentionFlowExplainerConfig(AttnExplainerConfig):
    type: Literal["attn/attention_flow"] = "attn/attention_flow"
    module_path: str | None = (
        "atria_insights.explainers._attn._base_explainer.AttentionFlowExplainer"
    )


AttnExplainerConfigType = Annotated[
    RawAttentionExplainerConfig
    | AttentionRolloutExplainerConfig
    | AttentionFlowExplainerConfig,
    Field(discriminator="type"),
]
