#!/bin/bash -l

# # task_pipeline=trainer/image_classification \
# uv run python \
#     -m atria_ml.task_pipelines.run \
#     task_pipeline=trainer/image_classification \
#     dataset@data_pipeline.dataset=cifar10/default \
#     model_pipeline.model.model_name=resnet18 \
#     model_pipeline.runtime_transforms.train.transforms.resize.params.size=[32,32] \
#     model_pipeline.runtime_transforms.evaluation.transforms.resize.params.size=[32,32] \
#     data_pipeline.dataloader_config.train_batch_size=256 \
#     data_pipeline.dataloader_config.eval_batch_size=256 \
#     data_pipeline.dataloader_config.num_workers=8 \
#     experiment_name=train_cifar10_resnet50 $@

# task_pipeline=trainer/image_classification \
uv run python \
    -m atria_ml.task_pipelines.run \
    task_pipeline=trainer/image_classification \
    dataset@data_pipeline.dataset=cifar10/default \
    model@model_pipeline.model=cifar10_models/resnet18 \
    data_transform@model_pipeline.runtime_transforms.train=cifar10 \
    model_pipeline.runtime_transforms.train.train=True \
    data_transform@model_pipeline.runtime_transforms.evaluation=cifar10 \
    optimizer@training_engine.optimizer=sgd \
    training_engine.with_amp=True \
    training_engine.optimizer.lr=0.1 \
    training_engine.optimizer.momentum=0.9 \
    training_engine.optimizer.weight_decay=5e-4 \
    data_pipeline.dataloader_config.train_batch_size=128 \
    data_pipeline.dataloader_config.eval_batch_size=128 \
    experiment_name=train_cifar10_resnet50_v3 $@