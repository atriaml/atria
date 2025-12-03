import os
from pathlib import Path

import hydra
from atria_core.logger import get_logger
from atria_core.logger.logger import LoggerBase
from atria_core.utilities.yaml_resolvers import *  # noqa this is needed to register the resolvers
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

logger = get_logger(__name__)


def initialize_and_run(
    local_rank: int, config: DictConfig, hydra_config: DictConfig
) -> None:
    from atria_core.utilities.pydantic import pydantic_parser
    from hydra_zen import instantiate

    from atria_ml.training.engines.utilities import RunConfig

    hydra_config = HydraConfig.get()
    LoggerBase().log_file_path = (
        Path(hydra_config.runtime.output_dir) / f"{hydra_config.job.name}.log"
    )
    LoggerBase().rank = local_rank
    config.output_dir = hydra_config.runtime.output_dir
    config._zen_exclude.append("package")  # exclude atria base config args
    config._zen_exclude.append("version")  # exclude atria base config args

    logger.info(f"Initializing task pipeline: {config.pipeline_name}")
    pipeline = instantiate(config, _convert_="object", _target_wrapper_=pydantic_parser)
    pipeline.build(
        local_rank=local_rank,
        experiment_name=config.experiment_name,
        run_config=RunConfig(data=config),
    )
    return pipeline.run()


@hydra.main(version_base=None, config_path="../conf", config_name="__atria__")
def app(config: DictConfig) -> None:
    hydra_config = HydraConfig.get()
    if config.n_devices > 1:
        import ignite.distributed as idist

        # we run the torch distributed environment with spawn if we have all the gpus on the same script
        # such as when we set --gpus-per-task=N
        ntasks = int(os.environ["SLURM_NTASKS"]) if "SLURM_JOBID" in os.environ else 1
        if ntasks == 1:
            job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else 0
            port = (int(job_id) + 10007) % 16384 + 49152
            logger.info(f"Starting distributed training on port: [{port}]")
            with idist.Parallel(
                backend=config.backend,
                nproc_per_node=config.n_devices,
                master_port=port,
            ) as parallel:
                return parallel.run(initialize_and_run, config, hydra_config)
        elif ntasks == int(config.n_devices):
            num_local_gpus = int(os.getenv("SLURM_GPUS_ON_NODE", 1))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, range(num_local_gpus))
            )
            with idist.Parallel(backend=config.backend) as parallel:
                return parallel.run(initialize_and_run, config, hydra_config)
        else:
            raise ValueError(
                f"Your slurm tasks do not match the number of required devices [{ntasks}!={config.n_devices}]."
            )
    else:
        try:
            initialize_and_run(0, config, hydra_config)
        except Exception as e:
            logger.exception(e)


if __name__ == "__main__":
    app()
