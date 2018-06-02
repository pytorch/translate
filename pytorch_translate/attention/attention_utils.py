#!/usr/bin/env python3

import torch


def create_src_lengths_mask(batch_size, src_lengths):
    """
    Generate boolean mask to prevent attention beyond the end of source

    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths

    Outputs:
      [batch_size, max_src_len]
    """
    max_srclen = src_lengths.max()
    src_indices = torch.arange(
        0,
        max_srclen,
    ).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_srclen)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(
        batch_size,
        max_srclen,
    )
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()
