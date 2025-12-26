# Setup guide for running CIFAR-10 examples explanation experiments

## Train the model

```bash
train.py --dataset_name cifar10/default --model_name resnet50 --max_epochs 50 --batch-size 128 --optim sgd --lr 0.01
```

## Explain the model predictions

```bash
explain.py --dataset_name cifar10/default --checkpoint_path <path_to_trained_model_checkpoint> --explainer_name saliency --batch_size 32
```
