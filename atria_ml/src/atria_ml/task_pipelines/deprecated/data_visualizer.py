from pathlib import Path

import hydra
from atria_core.logger.logger import LoggerBase, get_logger
from omegaconf import DictConfig

logger = get_logger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="task_pipeline/data_visualizer",
)
def app(config: DictConfig) -> None:
    from atria_ml.task_pipelines._data_visualizer import DataVisualizer  # noqa
    from atria_core.utilities.pydantic import pydantic_parser  # noqa
    from hydra.core.hydra_config import HydraConfig
    from hydra_zen import instantiate  # noqa

    hydra_config = HydraConfig.get()
    LoggerBase().log_file_path = (
        Path(hydra_config.runtime.output_dir) / f"{hydra_config.job.name}.log"
    )
    try:
        logger.info("Initializing DataVisualizer...")
        data_visualizer: DataVisualizer = instantiate(
            config, _convert_="object", _target_wrapper_=pydantic_parser
        )
        data_visualizer.build()
        data_visualizer.run()
    except Exception as e:
        logger.exception(f"Failed to instantiate DataVisualizer: {e}")


if __name__ == "__main__":
    app()
