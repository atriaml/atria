import enum

from pydantic import BaseModel


class ExplanationTargetStrategy(str, enum.Enum):
    predicted = "predicted"
    ground_truth = "ground_truth"
    all = "all"


class ExplainableModelPipelineConfig(BaseModel):
    explanation_target_strategy: ExplanationTargetStrategy = (
        ExplanationTargetStrategy.predicted
    )


config = ExplainableModelPipelineConfig()
print(config.model_dump())
