import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

# def get_extended_attention_mask(
#     self,
#     attention_mask: torch.Tensor,
#     input_shape: tuple[int],
#     device: torch.device = None,
#     dtype: torch.dtype = None,
# ) -> torch.Tensor:
#     """
#     Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

#     Arguments:
#         attention_mask (`torch.Tensor`):
#             Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
#         input_shape (`tuple[int]`):
#             The shape of the input to the model.

#     Returns:
#         `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
#     """
#     if dtype is None:
#         dtype = self.dtype

#     if not (attention_mask.dim() == 2 and self.config.is_decoder):
#         # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
#         if device is not None:
#             warnings.warn(
#                 "The `device` argument is deprecated and will be removed in v5 of Transformers.",
#                 FutureWarning,
#             )
#     # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#     # ourselves in which case we just need to make it broadcastable to all heads.
#     if attention_mask.dim() == 3:
#         extended_attention_mask = attention_mask[:, None, :, :]
#     elif attention_mask.dim() == 2:
#         # Provided a padding mask of dimensions [batch_size, seq_length]
#         # - if the model is a decoder, apply a causal mask in addition to the padding mask
#         # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         if self.config.is_decoder:
#             extended_attention_mask = (
#                 ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
#                     input_shape, attention_mask, device
#                 )
#             )
#         else:
#             extended_attention_mask = attention_mask[:, None, None, :]
#     else:
#         raise ValueError(
#             f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
#         )

#     # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
#     # masked positions, this operation will create a tensor which is 0.0 for
#     # positions we want to attend and the dtype's smallest value for masked positions.
#     # Since we are adding it to the raw scores before the softmax, this is
#     # effectively the same as removing these entirely.
#     extended_attention_mask = extended_attention_mask.to(
#         dtype=dtype
#     )  # fp16 compatibility
#     extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
#     return extended_attention_mask


def _resolve_attention_mask(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    use_sdpa_attention_masks: bool = True,
) -> torch.Tensor | None:
    input_shape = token_embeddings.size()[:-1]
    batch_size, seq_length = input_shape
    device = token_embeddings.device
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length), device=device)

    # Expand the attention mask
    if use_sdpa_attention_masks and attention_mask.dim() == 2:
        # Expand the attention mask for SDPA.
        # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
        return _prepare_4d_attention_mask_for_sdpa(
            attention_mask, token_embeddings.dtype, tgt_len=seq_length
        )
    else:
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        return self.get_extended_attention_mask(attention_mask, input_shape)


def _convert_head_mask_to_5d(
    head_mask: torch.Tensor, num_hidden_layers: int, dtype: torch.dtype
) -> torch.Tensor:
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = (
            head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        )  # We can specify head_mask for each layer
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(
        dtype=dtype
    )  # switch to float if need + fp16 compatibility
    return head_mask


def _resolve_head_mask(
    head_mask: torch.Tensor | None,
    num_hidden_layers: int,
    dtype: torch.dtype,
    is_attention_chunked: bool = False,
) -> torch.Tensor | None:
    if head_mask is not None:
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers, dtype=dtype)
        if is_attention_chunked is True:
            return head_mask.unsqueeze(-1)
        return head_mask
    return head_mask


def _pad_to_max_len(
    input_ids=None,
    bbox=None,
    attention_mask=None,
    token_type_ids=None,
    labels=None,
    position_ids=None,
    valid_span=None,
    max_len=512,
    pad_token_id=1,
):
    import torch
    import torch.nn.functional as F

    if input_ids is not None:
        seq_len = input_ids.shape[1]
    else:
        return (
            input_ids,
            bbox,
            attention_mask,
            token_type_ids,
            labels,
            position_ids,
            valid_span,
        )

    pad_len = max_len - seq_len
    if pad_len <= 0:
        return (
            input_ids,
            bbox,
            attention_mask,
            token_type_ids,
            labels,
            position_ids,
            valid_span,
        )

    input_ids = F.pad(input_ids, (0, pad_len), value=pad_token_id)

    # attention_mask: pad with 0 so padded tokens are ignored
    if attention_mask is not None:
        attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

    # token_type_ids: pad with 0
    if token_type_ids is not None:
        token_type_ids = F.pad(token_type_ids, (0, pad_len), value=0)

    # bbox: pad with [0, 0, 0, 0] boxes -> pad the sequence dim (dim=1), not the coordinate dim
    if bbox is not None:
        pad_boxes = bbox.new_zeros(bbox.shape[0], pad_len, bbox.shape[2])
        bbox = torch.cat([bbox, pad_boxes], dim=1)

    # labels: pad with -100 so CrossEntropyLoss ignores them
    if labels is not None:
        labels = F.pad(labels, (0, pad_len), value=-100)

    if position_ids is not None:
        position_ids = F.pad(position_ids, (0, pad_len), value=0)

    if valid_span is not None:
        seq_length = valid_span.shape[1]
        valid_span_padded = torch.zeros(
            valid_span.shape[0],
            max_len,
            max_len,
            dtype=torch.bool,
            device=valid_span.device,
        )
        valid_span_padded[:, :seq_length, :seq_length] = valid_span
        valid_span_padded[:, seq_length:, seq_length:] = False
        valid_span = valid_span_padded

    return (
        input_ids,
        bbox,
        attention_mask,
        token_type_ids,
        labels,
        position_ids,
        valid_span,
    )
