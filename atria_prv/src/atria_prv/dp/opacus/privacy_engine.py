import warnings
from itertools import chain

import torch
from atria_logger import get_logger
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader, default_collate, switch_generator
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample import AbstractGradSampleModule, GradSampleHooks
from opacus.optimizers import DPOptimizer
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch import nn, optim
from torch.distributed._composable.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.dataloader import _collate_fn_t

logger = get_logger(__name__)


class _DPDataLoader(DPDataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        sample_rate: float,
        collate_fn: _collate_fn_t | None = None,
        drop_last: bool = False,
        generator=None,
        distributed: bool = False,
        batch_first: bool = True,
        rand_on_empty: bool = False,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.distributed = distributed

        if distributed:
            batch_sampler = DistributedUniformWithReplacementSampler(
                total_size=len(dataset),  # type: ignore[assignment, arg-type]
                sample_rate=sample_rate,
                generator=generator,
            )
        else:
            batch_sampler = UniformWithReplacementSampler(
                num_samples=len(dataset),  # type: ignore[assignment, arg-type]
                sample_rate=sample_rate,
                generator=generator,
            )

        if collate_fn is None:
            collate_fn = default_collate

        if drop_last:
            logger.warning(
                "Ignoring drop_last as it is not compatible with DPDataLoader."
            )

        DataLoader.__init__(
            self,
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            generator=generator,
            **kwargs,
        )

    @classmethod
    def from_data_loader(
        cls,
        data_loader: DataLoader,
        *,
        distributed: bool = False,
        generator=None,
        batch_first: bool = True,
        rand_on_empty: bool = False,
        sample_rate: float | None = None,
    ):
        if isinstance(data_loader.dataset, IterableDataset):
            raise ValueError("Uniform sampling is not supported for IterableDataset")

        sample_rate = sample_rate or 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        logger.info(f"Sample rate = {sample_rate}")
        logger.info(f"Expected batch size = {expected_batch_size}")

        return cls(
            dataset=data_loader.dataset,
            sample_rate=sample_rate,
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            generator=generator if generator else data_loader.generator,
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
            distributed=distributed,
            batch_first=batch_first,
            rand_on_empty=rand_on_empty,
        )


class _PrivacyEngine(PrivacyEngine):
    def state_dict(self):
        return {"privacy_accountant": self.accountant.state_dict()}

    def load_state_dict(self, state_dict):
        self.accountant.load_state_dict(state_dict["privacy_accountant"])

    def _prepare_data_loader(
        self,
        data_loader: DataLoader,
        *,
        poisson_sampling: bool,
        distributed: bool,
        batch_first: bool = True,
        rand_on_empty: bool = False,
        sample_rate: float | None = None,
    ) -> DataLoader:

        if self.dataset is None:
            self.dataset = data_loader.dataset
        elif self.dataset != data_loader.dataset:
            warnings.warn(  # noqa: B028
                f"PrivacyEngine detected new dataset object. "
                f"Was: {self.dataset}, got: {data_loader.dataset}. "
                f"Privacy accounting works per dataset, please initialize "
                f"new PrivacyEngine if you're using different dataset. "
                f"You can ignore this warning if two datasets above "
                f"represent the same logical dataset"
            )

        if poisson_sampling:
            return _DPDataLoader.from_data_loader(
                data_loader,
                generator=self.secure_rng,
                distributed=distributed,
                batch_first=batch_first,
                rand_on_empty=rand_on_empty,
                sample_rate=sample_rate,
            )
        elif self.secure_mode:
            return switch_generator(data_loader=data_loader, generator=self.secure_rng)
        else:
            return data_loader

    def make_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        criterion=nn.CrossEntropyLoss(),  # Added deafult for backward compatibility
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: float | list[float],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        wrap_model: bool = True,
        rand_on_empty: bool = False,
        sample_rate: float | None = None,
        **kwargs,
    ) -> (
        tuple[AbstractGradSampleModule | GradSampleHooks, DPOptimizer, DataLoader]
        | tuple[
            AbstractGradSampleModule | GradSampleHooks,
            DPOptimizer,
            DPLossFastGradientClipping,
            DataLoader,
        ]
    ):
        if noise_generator and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        # compare module parameter with optimizer parameters
        model_parameters = set(module.parameters())
        for p in chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        ):
            if p not in model_parameters:
                raise ValueError(
                    "Module parameters are different than optimizer Parameters"
                )

        distributed = isinstance(module, (DPDDP, DDP, FSDPModule))

        module = self._prepare_model(
            module,
            batch_first=batch_first,
            max_grad_norm=max_grad_norm,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
            wrap_model=wrap_model,
        )
        if poisson_sampling:
            module.forbid_grad_accumulation()

        if sample_rate is None:
            sample_rate = 1 / len(data_loader)
            expected_batch_size = int(len(data_loader.dataset) * sample_rate)
        else:
            expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        data_loader = self._prepare_data_loader(
            data_loader,
            distributed=distributed,
            poisson_sampling=poisson_sampling,
            batch_first=batch_first,
            rand_on_empty=rand_on_empty,
            sample_rate=sample_rate,
        )

        # expected_batch_size is the *per worker* batch size
        if distributed:
            world_size = torch.distributed.get_world_size()
            expected_batch_size /= world_size

        optimizer = self._prepare_optimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
            **kwargs,
        )

        optimizer.attach_step_hook(
            self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
        )
        if "ghost" in grad_sample_mode:
            criterion = self._prepare_criterion(
                module=module,
                optimizer=optimizer,
                criterion=criterion,
                loss_reduction=loss_reduction,
                **kwargs,
            )

            return module, optimizer, criterion, data_loader

        return module, optimizer, data_loader
