from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def _generate_word_level_targets(word_ids_per_sample):
    targets_per_sample = []
    last_word_id = None
    for idx in range(word_ids_per_sample.shape[0]):
        if (
            word_ids_per_sample[idx] != -100
            and word_ids_per_sample[idx] != last_word_id
        ):
            targets_per_sample.append(idx)
        last_word_id = word_ids_per_sample[idx]
    return targets_per_sample


# def _generate_word_level_targets(
#     token_labels_per_sample: torch.Tensor,
#     predicted_token_labels_per_sample: torch.Tensor,
#     word_ids_per_sample: torch.Tensor,
#     percent_other_labels_kept: float = 0.0,
#     max_targets: int | None = None,
#     other_label_idx: int = 0,
#     seed: int = 0,
# ):
#     import torch

#     # find the tokens that are either predicted as Other or have ground-truth of other besides the padding labels
#     other_labels_mask = token_labels_per_sample == other_label_idx
#     other_labels_mask |= predicted_token_labels_per_sample == other_label_idx
#     other_labels_mask &= token_labels_per_sample != -100

#     # now extract indices of these other label tokens
#     other_labels_indices = other_labels_mask.nonzero().flatten()

#     # we randomly shuffle the indices of other labels
#     rand_other_labels_indices = other_labels_indices[
#         torch.randperm(
#             other_labels_indices.shape[0],
#             device=other_labels_indices.device,
#             # this seeding is necessary so that always the same targets for each sample are generated across methods/runs
#             generator=torch.Generator(other_labels_indices.device).manual_seed(seed),
#         )
#     ]

#     # extract (1-percent_other_labels_kept) of Other (O) labels to ignore
#     other_labels_ignored_indices = rand_other_labels_indices[
#         int(rand_other_labels_indices.shape[0] * percent_other_labels_kept) :
#     ]

#     targets = []
#     target_word_ids = []
#     last_word_id = None
#     for token_id in range(word_ids_per_sample.shape[0]):
#         if (
#             word_ids_per_sample[token_id] != -100
#             and word_ids_per_sample[token_id] != last_word_id
#             and token_id not in other_labels_ignored_indices
#         ):
#             targets.append(
#                 (token_id, predicted_token_labels_per_sample[token_id].item())
#             )
#             target_word_ids.append(word_ids_per_sample[token_id].item())
#         last_word_id = word_ids_per_sample[token_id]

#     # get N% of the total targets for final evaluation
#     random_indices = torch.randperm(
#         len(targets),
#         # this seeding is necessary so that always the same targets for each sample are generated across methods/runs
#         generator=torch.Generator().manual_seed(seed),
#     ).tolist()

#     # filter tokens
#     def filter(arr):
#         # rearrange indices randomly
#         arr = [arr[idx] for idx in random_indices]

#         if max_targets is not None:
#             arr = arr[:max_targets]

#         # take first max_target targets
#         return arr

#     targets = filter(targets)
#     target_word_ids = filter(target_word_ids)

#     # sanity check
#     assert len(targets) == len(target_word_ids)
#     return targets, target_word_ids
