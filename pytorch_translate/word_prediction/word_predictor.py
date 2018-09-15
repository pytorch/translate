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
        topk_labels_per_source_token=None,
        use_self_attention=False,
    ):
        super().__init__()
        self.encoder_output_dim = encoder_output_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.topk_labels_per_source_token = topk_labels_per_source_token
        self.use_self_attention = use_self_attention

        if self.use_self_attention:
            self.init_layer = nn.Linear(encoder_output_dim, encoder_output_dim)
            self.attn_layer = nn.Linear(2 * encoder_output_dim, 1)
            self.hidden_layer = nn.Linear(2 * encoder_output_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            self.hidden_layer = nn.Linear(encoder_output_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_output):
        # [source_length, batch_size, encoder_output_dim]
        encoder_hiddens, *_ = encoder_output
        assert encoder_hiddens.dim()

        if self.use_self_attention:
            # [batch_size, hidden_dim]
            init_state = self._get_init_state(encoder_hiddens)
            # [source_length, batch_size, 1]
            attn_scores = self._attention(encoder_hiddens, init_state)
            # [batch_size, hidden_dim]
            attned_state = (encoder_hiddens * attn_scores).sum(0)

            pred_input = torch.cat([init_state, attned_state], 1)
            pred_hidden = F.relu(self.hidden_layer(pred_input))
            # [batch_size, vocab_size]
            logits = self.output_layer(pred_hidden)
        else:
            # [source_length, batch_size, hidden_dim]
            hidden = F.relu(self.hidden_layer(encoder_hiddens))
            # [batch_size, hidden_dim]
            mean_hidden = torch.mean(hidden, 0)
            max_hidden = torch.max(hidden, 0)[0]
            # [batch_size, vocab_size]
            logits = self.output_layer(mean_hidden + max_hidden)

        return logits

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

    def get_topk_predicted_tokens(self, net_output, src_tokens, log_probs: bool):
        """
        Get self.topk_labels_per_source_token top predicted words for vocab
        reduction (per source token).
        """
        assert (
            isinstance(self.topk_labels_per_source_token, int)
            and self.topk_labels_per_source_token > 0
        ), "topk_labels_per_source_token must be a positive int, or None"

        # number of labels to predict for each example in batch
        k = src_tokens.size(1) * self.topk_labels_per_source_token
        # [batch_size, vocab_size]
        probs = self.get_normalized_probs(net_output, log_probs)
        _, topk_indices = torch.topk(probs, k, dim=1)

        return topk_indices
