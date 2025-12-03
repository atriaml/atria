#!/bin/bash -l

uv run python \
    -m atria_ml.task_pipelines.run \
    task_pipeline=trainer/image_classification \
    dataset@data_pipeline.dataset=tobacco3482/default \
    model_pipeline.model.timm_name=resnet50 \
    data_pipeline.dataloader_config.train_batch_size=64 \
    data_pipeline.dataloader_config.eval_batch_size=64 \
    data_pipeline.dataloader_config.num_workers=8 \
    experiment_name=train_tobacco3482_resnet50 \
    test_checkpoint=/mnt/hephaistos/projects/atria_agent/atria/atria_ml/outputs/trainer/image_classification/train_tobacco3482_resnet50/checkpoints/checkpoint_100.pt \
    $@
