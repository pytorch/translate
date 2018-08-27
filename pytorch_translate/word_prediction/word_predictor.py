#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordPredictor(nn.Module):
    def __init__(
        self,
        encoder_output_dim,
        hidden_dim,
        output_dim,
        prediction_vocab_reduction_topk=None,
    ):
        super().__init__()
        self.encoder_output_dim = encoder_output_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.init_layer = nn.Linear(encoder_output_dim, encoder_output_dim)
        self.attn_layer = nn.Linear(2 * encoder_output_dim, 1)
        self.hidden_layer = nn.Linear(2 * encoder_output_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.prediction_vocab_reduction_topk = prediction_vocab_reduction_topk

    def forward(self, encoder_output):
        # encoder_hiddens: [timestamp, batch_size, dim]
        encoder_hiddens, *_ = encoder_output
        assert encoder_hiddens.dim() == 3  # [T, B, EH]
        init_state = self._get_init_state(encoder_hiddens)  # [B, EH]

        attn_scores = self._attention(encoder_hiddens, init_state)  # [T, B, 1]
        attned_state = (encoder_hiddens * attn_scores).sum(0)  # [B, EH]

        pred_input = torch.cat([init_state, attned_state], 1)
        pred_hidden = F.relu(self.hidden_layer(pred_input))
        pred_logit = self.output_layer(pred_hidden)  # [batch, vocab]
        return pred_logit

    def _get_init_state(self, encoder_hiddens):
        x = torch.mean(encoder_hiddens, 0)
        x = F.relu(self.init_layer(x))
        return x

    def _attention(self, encoder_hiddens, init_state):
        init_state = init_state.unsqueeze(0).expand_as(encoder_hiddens)
        attn_input = torch.cat([init_state, encoder_hiddens], 2)
        attn_scores = F.relu(self.attn_layer(attn_input))
        attn_scores = F.softmax(attn_scores, 0)
        return attn_scores

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output  # [batch, vocab]
        if log_probs:
            return F.log_softmax(logits, dim=1)
        else:
            return F.softmax(logits, dim=1)

    def get_topk_predicted_tokens(self, net_output, log_probs: bool):
        """
        Get self.prediction_vocab_reduction_topk top predicted words for vocab
        reduction.
        """
        assert (
            isinstance(self.prediction_vocab_reduction_topk, int)
            and self.prediction_vocab_reduction_topk > 0
        ), "--prediction-vocab-reduction-topk must be a positive int, or None"
        probs = self.get_normalized_probs(net_output, log_probs)  # [batch, vocab]
        _, topk_indices = torch.topk(probs, self.prediction_vocab_reduction_topk, dim=1)
        return topk_indices
