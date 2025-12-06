# file: configs_example.py
import json
from typing import List

from hydra.utils import instantiate
from pydantic import BaseModel


# --------------------------
# Define nested Pydantic configs
# --------------------------
class OptimizerConfig(BaseModel):
    lr: float
    weight_decay: float


class ModelConfig(BaseModel):
    layers: int
    hidden_size: int
    optimizer: OptimizerConfig


class TrainConfig(BaseModel):
    epochs: int
    batch_size: int
    model: ModelConfig


class RunConfig(BaseModel):
    run_name: str
    train: TrainConfig
    extra_models: List[ModelConfig]


# --------------------------
# Helper: convert Pydantic to Hydra-style dict
# --------------------------
def pydantic_to_hydra(obj: BaseModel):
    """
    Recursively convert a Pydantic BaseModel into a dict suitable
    for Hydra instantiate (with _target_).
    """
    data = obj.model_dump()
    data["_target_"] = f"{obj.__class__.__module__}.{obj.__class__.__name__}"

    for k, v in data.items():
        if isinstance(v, BaseModel):
            data[k] = pydantic_to_hydra(v)
        elif isinstance(v, list):
            data[k] = [
                pydantic_to_hydra(i) if isinstance(i, BaseModel) else i for i in v
            ]
        elif isinstance(v, dict):
            data[k] = {
                kk: pydantic_to_hydra(vv) if isinstance(vv, BaseModel) else vv
                for kk, vv in v.items()
            }
    return data


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # 1️⃣ Create nested Pydantic config object
    cfg = RunConfig(
        run_name="experiment_1",
        train=TrainConfig(
            epochs=10,
            batch_size=32,
            model=ModelConfig(
                layers=4,
                hidden_size=128,
                optimizer=OptimizerConfig(lr=0.001, weight_decay=0.01),
            ),
        ),
        extra_models=[
            ModelConfig(
                layers=2,
                hidden_size=64,
                optimizer=OptimizerConfig(lr=0.01, weight_decay=0.001),
            ),
            ModelConfig(
                layers=3,
                hidden_size=256,
                optimizer=OptimizerConfig(lr=0.0001, weight_decay=0.1),
            ),
        ],
    )

    # 2️⃣ Dump to Hydra-style dict
    hydra_cfg = pydantic_to_hydra(cfg)
    print("Hydra-style dict:\n", json.dumps(hydra_cfg, indent=2))

    # 3️⃣ Save to JSON
    with open("config.json", "w") as f:
        json.dump(hydra_cfg, f, indent=2)

    # 4️⃣ Load from JSON
    with open("config.json") as f:
        loaded_cfg = json.load(f)

    # 5️⃣ Instantiate objects using Hydra
    instantiated_cfg = instantiate(loaded_cfg)
    print("\nInstantiated object:\n", instantiated_cfg)

    # 6️⃣ Check types
    assert instantiated_cfg.train.model.optimizer.lr == 0.001
    assert isinstance(instantiated_cfg.extra_models[0].optimizer, OptimizerConfig)
