import warnings

from atria_logger import get_logger
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader, default_collate, switch_generator
from opacus.utils.uniform_sampler import (
    DistributedUniformWithReplacementSampler,
    UniformWithReplacementSampler,
)
from torch.utils.data import DataLoader, Dataset
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
    ) -> DataLoader:

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)
        print("Sample rate = ", sample_rate)
        print("Expected batch size = ", expected_batch_size)
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
            )
        elif self.secure_mode:
            return switch_generator(data_loader=data_loader, generator=self.secure_rng)
        else:
            return data_loader
