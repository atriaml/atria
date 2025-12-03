import sys
from pathlib import Path

from atria_core.logger import get_logger
from atria_core.utilities import yaml_resolvers  # noqa

from atria_ml.registry import TASK_PIPELINE
from atria_ml.training.engines.utilities import RunConfig

logger = get_logger(__name__)


# def initialize_and_run(config: DictConfig, hydra_config: DictConfig) -> None:
#     from atria_core.utilities.pydantic import pydantic_parser
#     from hydra_zen import instantiate

#     from atria_ml.task_pipelines._trainer import Trainer
#     from atria_ml.training.engines.utilities import RunConfig

#     LoggerBase().log_file_path = (
#         Path(hydra_config.runtime.output_dir) / f"{hydra_config.job.name}.log"
#     )
#     config.output_dir = hydra_config.runtime.output_dir
#     config._zen_exclude.append("package")  # exclude atria base config args
#     config._zen_exclude.append("version")  # exclude atria base config args
#     atria_trainer: Trainer = instantiate(
#         config, _convert_="object", _target_wrapper_=pydantic_parser
#     )
#     atria_trainer.build(run_config=RunConfig(data=config))
#     return atria_trainer.run()


def app() -> None:
    from atria_core.logger import LoggerBase

    from atria_ml.task_pipelines._trainer import Trainer

    overrides = sys.argv[1:]
    logger.info("Loading Atria Trainer with overrides: %s", overrides)
    atria_trainer, _, config = TASK_PIPELINE.load_from_registry(
        "trainer/image_classification",
        overrides=overrides,
        search_pkgs=[
            "atria_models",
            "atria_datasets",
            "atria_transforms",
            "atria_metrics",
        ],
        return_config=True,
    )

    # Set the log file path for the logger
    output_dir = config.get("output_dir")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    LoggerBase().log_file_path = Path(config["output_dir"]) / "atria_trainer.log"

    # Initialize the Atria Trainer with the provided configuration
    atria_trainer: Trainer
    atria_trainer.build(local_rank=0, run_config=RunConfig(data=config))
    return atria_trainer.run()

    # try:
    #     from rich.pretty import pprint

    #     pprint(config.hydra)
    #     logger.info("Initializing Atria Trainer with config...")
    #     initialize_and_run(config, config.hydra)
    # except Exception as e:
    #     logger.exception(e)


if __name__ == "__main__":
    app()
