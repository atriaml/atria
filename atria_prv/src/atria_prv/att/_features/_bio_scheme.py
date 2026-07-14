from __future__ import annotations

import torch


class BioScheme:
    """Derives BIO-aware token masks (entity / span-start / span-continuation / other).

    Prefer :meth:`from_label_names`, which reads the actual ``B-``/``I-``/``O`` prefixes
    off the dataset's label list. :meth:`from_num_labels` is a fallback for datasets that
    only expose a label count, assuming label id ``0`` is ``O`` and non-zero ids pair up
    as odd=B-/even=I-.
    """

    def __init__(self, o_ids: set[int], b_ids: set[int], i_ids: set[int]) -> None:
        self._o_ids = torch.tensor(sorted(o_ids), dtype=torch.long)
        self._b_ids = torch.tensor(sorted(b_ids), dtype=torch.long)
        self._i_ids = torch.tensor(sorted(i_ids), dtype=torch.long)
        self._entity_ids = torch.cat([self._b_ids, self._i_ids])

    @classmethod
    def from_label_names(cls, label_names: list[str]) -> BioScheme:
        o_ids, b_ids, i_ids = set(), set(), set()
        for idx, name in enumerate(label_names):
            if name.startswith("B-"):
                b_ids.add(idx)
            elif name.startswith("I-"):
                i_ids.add(idx)
            else:
                o_ids.add(idx)
        return cls(o_ids=o_ids, b_ids=b_ids, i_ids=i_ids)

    @classmethod
    def from_num_labels(cls, num_labels: int, o_label_id: int = 0) -> BioScheme:
        non_o = [i for i in range(num_labels) if i != o_label_id]
        b_ids = set(non_o[0::2])
        i_ids = set(non_o[1::2])
        return cls(o_ids={o_label_id}, b_ids=b_ids, i_ids=i_ids)

    def _isin_mask(self, labels: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        return torch.isin(labels, ids.to(labels.device))

    def all_mask(self, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        return labels != ignore_index

    def entity_mask(self, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        return self._isin_mask(labels, self._entity_ids) & self.all_mask(
            labels, ignore_index
        )

    def span_start_mask(self, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        return self._isin_mask(labels, self._b_ids) & self.all_mask(
            labels, ignore_index
        )

    def span_continuation_mask(
        self, labels: torch.Tensor, ignore_index: int
    ) -> torch.Tensor:
        return self._isin_mask(labels, self._i_ids) & self.all_mask(
            labels, ignore_index
        )

    def other_mask(self, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
        return self._isin_mask(labels, self._o_ids) & self.all_mask(
            labels, ignore_index
        )

    def splits(
        self, labels: torch.Tensor, ignore_index: int
    ) -> dict[str, torch.Tensor]:
        return {
            "all": self.all_mask(labels, ignore_index)
            # "entity": self.entity_mask(labels, ignore_index),
            # "span_start": self.span_start_mask(labels, ignore_index),
            # "span_cont": self.span_continuation_mask(labels, ignore_index),
            # "other": self.other_mask(labels, ignore_index),
        }
