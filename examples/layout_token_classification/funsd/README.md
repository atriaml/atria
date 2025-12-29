# Setup guide for running CIFAR-10 examples explanation experiments

## Train the model

```bash
python train.py --dataset_name funsd --model_name bert-base-uncased --max_epochs 50 --batch-size 32 --optim adamw --lr 2e-5
```

```bash
python train.py --dataset_name funsd --model_name roberta-base-uncased --max_epochs 50 --batch-size 32 --optim adam --lr 2e-5
```

```bash
python train.py --dataset_name funsd --model_name lilt-base --max_epochs 50 --batch-size 32 --optim adam --lr 2e-5
```

```bash
python train.py --dataset_name funsd --model_name layoutlmv3-base --max_epochs 50 --batch-size 32 --optim adam --lr 2e-5
```

## Explain the model predictions

```bash
explain.py --dataset_name funsd --checkpoint_path <path_to_trained_model_checkpoint> --explainer_name saliency --batch_size 32
```
