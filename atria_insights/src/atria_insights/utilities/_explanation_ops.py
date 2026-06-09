import torch
from atria_logger import get_logger

from atria_insights.utilities._explanation_units import (
    ImageExplanationUnit,
    TextExplanationUnit,
    TextLayoutExplanationUnit,
    TextPositionExplanationUnit,
)

logger = get_logger(__name__)


def _normalize_explanations(
    explanations: tuple[torch.Tensor, ...],
    shift_0_to_1: bool = True,
    outlier_perc: int = 2,
    debug: bool = False,
) -> tuple[torch.Tensor, ...]:
    from captum.attr._utils.visualization import _normalize_attr

    shapes = [exp.shape for exp in explanations]
    flat = torch.cat([exp.reshape(-1) for exp in explanations])

    if debug:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(flat.cpu().detach().numpy(), bins=100)
        axes[0].set_title("Before normalization")

    try:
        flat = _normalize_attr(flat.cpu(), sign="all", outlier_perc=outlier_perc).to(
            flat.device
        )
    except Exception:
        flat = torch.zeros_like(flat)

    if shift_0_to_1:
        flat = flat * 0.5 + 0.5

    if debug:
        axes[1].hist(flat.cpu().detach().numpy(), bins=100)
        axes[1].set_title("After normalization")
        plt.tight_layout()
        plt.show()

    sizes = [exp.numel() for exp in explanations]
    splits = flat.split(sizes)
    return tuple(s.reshape(shape) for s, shape in zip(splits, shapes, strict=True))


def _reduce_word_level_explanation_single(
    explanation: torch.Tensor,  # (seq_len,) or (seq_len, hidden)
    word_ids: torch.Tensor,  # (seq_len,)
    sequence_ids: torch.Tensor,  # (seq_len,)
) -> torch.Tensor:
    mask = word_ids != -100
    exp = explanation[mask]
    if exp.dim() > 1:
        exp = exp.sum(dim=-1)
    wid = word_ids[mask]
    sid = sequence_ids[mask]

    flat_key = sid * (wid.max() + 1) + wid
    unique_keys, inverse = flat_key.unique(return_inverse=True)
    num_groups = unique_keys.shape[0]

    sums = torch.zeros(num_groups, device=exp.device, dtype=exp.dtype)
    counts = torch.zeros(num_groups, device=exp.device, dtype=exp.dtype)
    sums.scatter_add_(0, inverse, exp)
    counts.scatter_add_(0, inverse, torch.ones_like(exp))
    return sums / counts


def _process_single_sample(
    feature_keys: set[str],
    sample_explanations: tuple[torch.Tensor, ...],
    word_ids: torch.Tensor,
    sequence_ids: torch.Tensor,
    context_text: list[str] | None,
) -> list[TextExplanationUnit | ImageExplanationUnit]:  # 1. reduce
    TEXT_EMBEDDING_KEYS = {
        "token_embeddings": TextExplanationUnit,
        "position_embeddings": TextPositionExplanationUnit,
        "layout_embeddings": TextLayoutExplanationUnit,
    }

    reduced = ()
    for key, exp in zip(feature_keys, sample_explanations, strict=True):
        logger.debug(f"[reduce] key={key} input shape={exp.shape}")
        if key == "image":
            reduced_exp = exp.sum(dim=0, keepdim=True)
        else:
            reduced_exp = _reduce_word_level_explanation_single(
                exp, word_ids, sequence_ids
            )
        logger.debug(f"[reduce] key={key} output shape={reduced_exp.shape}")
        reduced += (reduced_exp,)

    # 2. normalize
    logger.debug(f"[normalize] input shapes={[r.shape for r in reduced]}")
    reduced = _normalize_explanations(reduced)
    logger.debug(f"[normalize] output shapes={[r.shape for r in reduced]}")

    # 3. build units
    units = []
    context_len = len(context_text) if context_text is not None else 0
    logger.debug(f"[build] context_len={context_len}")

    for key, value in zip(feature_keys, reduced, strict=True):
        logger.debug(f"[build] key={key} value shape={value.shape}")
        if key in TEXT_EMBEDDING_KEYS:
            attribution = value[context_len:] if context_len else value
            context_attribution = value[:context_len] if context_len else None
            logger.debug(
                f"[build] key={key} attribution shape={attribution.shape} context_attribution shape={context_attribution.shape if context_attribution is not None else None}"
            )
            units.append(
                TEXT_EMBEDDING_KEYS[key](
                    attribution=attribution, context_attribution=context_attribution
                )
            )
        elif key == "image":
            logger.debug(f"[build] key={key} attribution shape={value.shape}")
            units.append(ImageExplanationUnit(attribution=value))

    return units
