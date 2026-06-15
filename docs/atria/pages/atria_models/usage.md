---
title: Usage
---

# Usage

## Load a registered pipeline config

```python
from atria_models.api import load_model_pipeline_config

config = load_model_pipeline_config("image/classification/timm")
```

## Instantiate and run a forward pass

```python
pipeline = config.build(labels=dataset.labels)
pipeline.model.eval()

# Pass a batch of tensors
output = pipeline(pixel_values=batch["pixel_values"])
logits = output.logits
```

## Create a pipeline with a custom backbone

```python
from atria_models.core.model_pipelines import ImageClassificationPipeline
from atria_models.core.model_pipelines._common import ModelConfig

pipeline = ImageClassificationPipeline(
    config=ImageClassificationPipeline.Config(
        model=ModelConfig(
            builder_type="timm",
            model_name_or_path="vit_base_patch16_224",
            model_kwargs={"pretrained": True},
        )
    ),
    labels=dataset.labels,
)
```

## Push a trained pipeline to the hub

```python
artifact = pipeline.ops.to_snapshot(checkpoint_path="./runs/best.pt")
pipeline.ops.push_to_hub(artifact, hub_model_name="my-org/vit-cifar10")
```

## Pull from the hub

```python
from atria_models.api import load_model_pipeline_config

config = load_model_pipeline_config("my-org/vit-cifar10", from_hub=True)
pipeline = config.build(labels=...)
```
