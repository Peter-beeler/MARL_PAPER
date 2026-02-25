"""
Batched sequence log-probability computation.
"""

import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence


def compute_batch_sequence_log_prob(
    model,
    prompt_input_ids_list: List[torch.Tensor],
    generated_ids_list: List[torch.Tensor],
    device: torch.device,
    pad_token_id: int = None,
    need_grad: bool = False
) -> torch.Tensor:
    """
    Compute log-probabilities for generated sequences conditioned on prompts.

    Args:
        model: Language model.
        prompt_input_ids_list: List of prompt token ID tensors.
        generated_ids_list: List of generated token ID tensors.
        device: Target device.
        pad_token_id: Pad token ID (inferred from model config if None).
        need_grad: Whether to compute gradients.

    Returns:
        Tensor of shape [batch_size] with sum log-probs for the generated part.
    """
    if pad_token_id is None:
        actual_model = model.module if hasattr(model, 'module') else model
        if hasattr(actual_model.config, 'pad_token_id'):
            pad_token_id = actual_model.config.pad_token_id

    if pad_token_id is None:
        raise ValueError("pad_token_id must be provided or available in model config")

    full_sequences = []
    prompt_lens = []
    gen_lens = []

    for prompt_ids, gen_ids in zip(prompt_input_ids_list, generated_ids_list):
        p_ids = prompt_ids.to(device).view(-1)
        p_ids = p_ids[p_ids != pad_token_id]

        g_ids = gen_ids.to(device).view(-1)
        g_ids = g_ids[g_ids != pad_token_id]

        prompt_lens.append(len(p_ids))
        gen_lens.append(len(g_ids))
        full_sequences.append(torch.cat([p_ids, g_ids]))

    batch_input_ids = pad_sequence(full_sequences, batch_first=True, padding_value=pad_token_id)
    attention_mask = (batch_input_ids != pad_token_id).long()

    def forward_pass():
        outputs = model(input_ids=batch_input_ids, attention_mask=attention_mask)
        return outputs.logits

    if need_grad:
        logits = forward_pass()
    else:
        with torch.no_grad():
            logits = forward_pass()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch_input_ids[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)
    token_log_probs = -loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size())

    action_mask = torch.zeros_like(token_log_probs)
    for i, (p_len, g_len) in enumerate(zip(prompt_lens, gen_lens)):
        start_idx = max(0, p_len - 1)
        end_idx = start_idx + g_len
        if g_len > 0:
            action_mask[i, start_idx:end_idx] = 1.0

    masked_log_probs = token_log_probs * action_mask
    seq_log_prob_sum = masked_log_probs.sum(dim=1)

    return seq_log_prob_sum
