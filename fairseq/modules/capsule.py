from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, num_out_caps, num_in_caps, dim_in_caps, dim_out_caps, num_iterations=3,
                 share_route_weights_for_in_caps=False):
        super(CapsuleLayer, self).__init__()

        self.num_out_caps = num_out_caps

        self.num_iterations = num_iterations
        assert num_iterations > 1, "num_iterations must at least be 1."

        self.share_route_weights_for_in_caps = share_route_weights_for_in_caps
        if share_route_weights_for_in_caps:
            self.route_weights = nn.Parameter(0.01 * torch.randn(num_out_caps, dim_in_caps, dim_out_caps))
        else:
            self.route_weights = nn.Parameter(0.01 * torch.randn(num_in_caps, num_out_caps, dim_in_caps, dim_out_caps))

    def __repr__(self):
        return super().__repr__() + "\n(routing_weights): {}".format(self.route_weights.size())

    @staticmethod
    def squash(tensor, dim=-1, eps=6e-8):
        tensor = tensor.float()
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        norm = torch.sqrt(squared_norm) + eps
        return scale * tensor / norm

    def forward(self, inputs_u, inputs_mask):
        """
        Args:
            inputs_u: Tensor. [batch_size, num_in_caps, dim_in_caps]
            inputs_mask: Tensor. [batch_size, num_in_caps]

        Returns: Tensor. [batch_size, num_out_caps, dim_out_caps]

        """
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        # Compute u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        if self.share_route_weights_for_in_caps:
            # priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, None, :, :, :]).squeeze(-2)
            inputs_u_r = inputs_u.view(batch_size * num_in_caps, dim_in_caps)
            route_weight_r = self.route_weights.transpose(0, 1).reshape(dim_in_caps, -1)
            priors_u_hat = inputs_u_r @ route_weight_r
            priors_u_hat = priors_u_hat.view(batch_size, num_in_caps, self.num_out_caps, -1)
        else:
            priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, :, :, :, :]).squeeze(-2)

        # Initialize logits
        # logits_b: [batch_size, num_in_caps, num_out_caps]
        logits_b = inputs_u.new_zeros(batch_size, num_in_caps, self.num_out_caps)

        # Routing
        for i in range(self.num_iterations):
            # probs: [batch_size, num_in_caps, num_out_caps]
            if inputs_mask is not None:
                logits_b = logits_b + inputs_mask.unsqueeze(-1) * -1e18
            probs_c = F.softmax(logits_b, dim=-1)
            # outputs_v: [batch_size, num_out_caps, dim_out_caps]
            outputs_v = self.squash((probs_c.unsqueeze(-1) * priors_u_hat).sum(dim=1))

            if i != self.num_iterations - 1:
                # delta_logits: [batch_size, num_in_caps, num_out_caps]
                delta_logits = (priors_u_hat * outputs_v.unsqueeze(1)).sum(dim=-1)
                logits_b = logits_b + delta_logits

        # outputs_v: [batch_size, num_out_caps, dim_out_caps]
        return outputs_v


class ContextualCapsuleLayer(CapsuleLayer):
    def __init__(self, num_out_caps, num_in_caps, dim_in_caps, dim_out_caps,
                 dim_context=None,
                 num_iterations=3,
                 share_route_weights_for_in_caps=False):
        super().__init__(num_out_caps, num_in_caps, dim_in_caps, dim_out_caps, num_iterations,
                         share_route_weights_for_in_caps)
        self.linear_u_hat = nn.Linear(dim_out_caps, dim_out_caps)
        self.linear_v = nn.Linear(dim_out_caps, dim_out_caps)
        self.linear_delta = nn.Linear(dim_out_caps, 1, False)

        self.dim_out_caps = dim_out_caps

        self.contextual = dim_context is not None
        if self.contextual:
            self.linear_c = nn.Linear(dim_context, dim_out_caps, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_u_hat.weight, 0, 0.001)
        nn.init.normal_(self.linear_v.weight, 0, 0.001)
        nn.init.normal_(self.linear_delta.weight, 0, 0.001)
        if self.contextual:
            nn.init.normal_(self.linear_c.weight, 0, 0.001)

    def compute_delta(self, priors_u_hat, outputs_v, contexts=None):
        """
        Args:
            priors_u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
            outputs_v: [batch_size, num_out_caps, dim_out_caps]
            contexts: [batch_size, dim_context]

        Returns: Tensor. [batch_size, num_in_caps, num_out_caps]

        """
        # [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        u = priors_u_hat
        v = outputs_v[:, None, :, :]
        # [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        delta = self.linear_u_hat(u) + self.linear_v(v)
        if self.contextual:
            c = contexts[:, None, None, :]
            delta = delta + self.linear_c(c)

        delta = F.tanh(delta)

        # [batch_size, num_in_caps, num_out_caps]
        delta = self.linear_delta(delta).squeeze(-1)  # [batch_size, num_in_caps, num_out_caps]
        delta = F.tanh(delta)
        return delta * (self.dim_out_caps ** -0.5)

    def compute_delta_sequence(self, priors_u_hat, outputs_v, contexts=None):
        """
        Args:
            priors_u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
            outputs_v: [batch_size, length, num_out_caps, dim_out_caps]
            contexts: [batch_size, length, dim_context]

        Returns: Tensor. [batch_size, length, num_in_caps, num_out_caps]

        """
        # [batch_size, length, num_in_caps, num_out_caps, dim_out_caps]
        u = priors_u_hat[:, None, :, :, :]
        v = outputs_v[:, :, None, :, :]
        # [batch_size, length, num_in_caps, num_out_caps, dim_out_caps]
        delta = self.linear_u_hat(u) + self.linear_v(v)

        # [batch, length, 1, 1, dim_context]
        c = contexts[:, :, None, None, :]
        # [batch, length, num_in_caps, num_out_caps, dim_out_caps]
        delta = delta + self.linear_c(c)

        delta = torch.tanh(delta)

        # [batch_size, length, num_in_caps, num_out_caps]
        delta = self.linear_delta(delta).squeeze(-1)
        delta = torch.tanh(delta)
        return delta * (self.dim_out_caps ** -0.5)

    def forward(self, inputs_u, inputs_mask, context=None):
        """
        Args:
            inputs_u: Tensor. [batch_size, num_in_caps, dim_in_caps]
            inputs_mask: Tensor. [batch_size, num_in_caps]
            context: [batch_size, dim_context]

        Returns: Tensor. [batch_size, num_out_caps, dim_out_caps]

        """
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        # Compute u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        if self.share_route_weights_for_in_caps:
            # priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, None, :, :, :]).squeeze(-2)
            inputs_u_r = inputs_u.view(batch_size * num_in_caps, dim_in_caps)
            route_weight_r = self.route_weights.transpose(0, 1).reshape(dim_in_caps, -1)
            priors_u_hat = inputs_u_r @ route_weight_r
            priors_u_hat = priors_u_hat.view(batch_size, num_in_caps, self.num_out_caps, -1)
        else:
            priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, :, :, :, :]).squeeze(-2)

        # Initialize logits
        # logits_b: [batch_size, num_in_caps, num_out_caps]
        logits_b = inputs_u.new_zeros(batch_size, num_in_caps, self.num_out_caps)

        # Routing
        for i in range(self.num_iterations):
            # probs: [batch_size, num_in_caps, num_out_caps]
            if inputs_mask is not None:
                logits_b = logits_b + inputs_mask.unsqueeze(-1) * -1e18
            probs_c = F.softmax(logits_b, dim=-1)
            # outputs_v: [batch_size, num_out_caps, dim_out_caps]
            outputs_v = self.squash((probs_c.unsqueeze(-1) * priors_u_hat).sum(dim=1))

            if i != self.num_iterations - 1:
                # delta_logits: [batch_size, num_in_caps, num_out_caps]
                delta_logits = self.compute_delta(priors_u_hat, outputs_v, context)
                logits_b = logits_b + delta_logits

        # outputs_v: [batch_size, num_out_caps, dim_out_caps]
        return outputs_v, probs_c

    def forward_sequence(self, inputs_u, inputs_mask, new_times=5, context_sequence=None, cache=None):
        """
        Args:
            inputs_u (torch.Tensor). [batch_size, num_in_caps, dim_in_caps]
            inputs_mask (torch.Tensor). [batch_size, num_in_caps]
            context_sequence (torch.Tensor) : [batch_size, length, dim_context]

        Returns: Tensor. [batch_size, length, num_out_caps, dim_out_caps]

        """
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        length = context_sequence.size(1)
        # Compute u_hat: [batch_size, num_in_caps, num_out_caps, dim_out_caps]
        if cache is not None:
            priors_u_hat = cache
        else:
            priors_u_hat = self.compute_caches(inputs_u)

        # Initialize logits
        # logits_b: [batch_size, length, num_in_caps, num_out_caps]
        logits_b = inputs_u.new_zeros(batch_size, length, num_in_caps, self.num_out_caps)
        # [batch, 1, num_in_caps, 1]
        if inputs_mask is not None:
            routing_mask = inputs_mask[:, None, :, None].expand_as(logits_b)
        # 添加同传相关的mask
        new_masked = inputs_u.new_zeros(length,num_in_caps).masked_fill(mask = torch.tril(inputs_u.new_ones(num_in_caps,length),diagonal=-new_times).transpose(0,1).bool(), value = torch.tensor(-np.inf,device=inputs_u.device))

        # Routing 
        for i in range(self.num_iterations):
            # probs: [batch_size, length, num_in_caps, num_out_caps]
            if inputs_mask is not None:
                logits_b = logits_b.masked_fill(routing_mask, float('-inf'))
            probs = logits_b[:,None,:,:,:].transpose(-1,1).squeeze(-1) + new_masked
            probs = torch.exp(probs[:,:,:,:,None].transpose(-1,1).squeeze(1))
            probs_c = (probs.sum(-1).unsqueeze(-1)+1e-8)
            probs_c = probs_c+6e-8
            probs_c = probs / probs_c

            # # [batch, num_out_caps, length,
            # _interm = probs_c.permute([0, 3, 1, 2]) @ prior_u_hat.transpose(1, 2))
            # outputs_v: [batch_size, length, num_out_caps, dim_out_caps]
            outputs_v = self.squash((probs_c.unsqueeze(-1) * priors_u_hat.unsqueeze(1)).sum(2))
            outputs_v = outputs_v.type_as(inputs_u)

            if i != self.num_iterations - 1:
                # delta_logits: [batch_size, length, num_in_caps, num_out_caps]
                delta_logits = self.compute_delta_sequence(
                    priors_u_hat, outputs_v, context_sequence
                )
                logits_b = logits_b + delta_logits

        # outputs_v: [batch_size, length, num_out_caps, dim_out_caps]
        return outputs_v, probs_c

    def compute_caches(self, inputs_u):
        batch_size, num_in_caps, dim_in_caps = inputs_u.size()
        if self.share_route_weights_for_in_caps:
            inputs_u_r = inputs_u.contiguous().view(batch_size * num_in_caps, dim_in_caps)
            route_weight_r = self.route_weights.transpose(0, 1).reshape(dim_in_caps, -1)
            priors_u_hat = inputs_u_r @ route_weight_r
            priors_u_hat = priors_u_hat.view(batch_size, num_in_caps, self.num_out_caps, -1)
        else:
            priors_u_hat = (inputs_u[:, :, None, None, :] @ self.route_weights[None, :, :, :,
                                                            :]).squeeze(-2)
        return priors_u_hat


class MultiInputPositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            size (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, size, hidden_size, dropout=0.1, inp_sizes=[]):
        super().__init__()
        self.total_input_size = size + sum(inp_sizes)
        self.w_1 = nn.Linear(self.total_input_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = nn.LayerNorm(self.total_input_size)

        # Save a little memory, by doing inplace.
        self.dropout_1 = nn.Dropout(dropout, inplace=False)
        self.relu = nn.ReLU(inplace=False)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, *x):
        inps = torch.cat(x, -1)
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(inps))))
        output = self.dropout_2(self.w_2(inter))
        return output + x[0]


class Generator(nn.Module):

    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=-1):
        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.proj = nn.Linear(self.hidden_size, self.n_words, bias=False)

        if shared_weight is not None:
            self.proj.linear.weight = shared_weight

    def _pad_2d(self, x):

        if self.padding_idx == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.padding_idx] = float('-inf')
            x_2d = x_2d + mask

            return x_2d.view(x_size)

    def forward(self, input, log_probs=True):
        """
        input == > Linear == > LogSoftmax
        """

        logits = self.proj(input)

        logits = self._pad_2d(logits)

        if log_probs:
            return F.log_softmax(logits, dim=-1,dtype=torch.float32)
        else:
            return F.softmax(logits, dim=-1)

class WordPredictor(nn.Module):
    def __init__(self, input_size, generator=None, **config):
        super().__init__()
        if generator is not None:
            self.generator = generator
        else:
            self.generator = Generator(n_words=config['n_tgt_vocab'],
                                             hidden_size=config['d_word_vec'],
                                             padding_idx=1)
        self.linear = nn.Linear(input_size, config['d_word_vec'])

    def forward(self, hiddens, logprob=True):
        logits = torch.tanh(self.linear(hiddens))
        return self.generator(logits, logprob)

def get_average_score(mask,New_type):
    mask = mask.type(New_type)
    scores = mask / mask.sum(-1, keepdim=True)
    scores = torch.where(torch.isnan(scores),
                         torch.zeros_like(scores),
                         scores)
    return scores

def convert_to_past_labels(labels,New_type=None, padding_idx=1):
    """
    Args:
        padding_idx:
        labels: [batch, seq_len]

    Returns:
        descending labels .
            [batch, seq_len, seq_len]
    """
    batch_size, seq_len = labels.size()
    seq_mask = labels.ne(padding_idx)

    # use upper triangle to masking in descending manner
    # [batch, seq_len, seq_len]
    step_mask = torch.tril(labels.new_ones(seq_len, seq_len), 0).byte()
    mask = step_mask.unsqueeze(0) * seq_mask.unsqueeze(1)
    tmp_mask = labels.eq(padding_idx)
    mask = mask.masked_fill(tmp_mask.unsqueeze(-1), 0)
    mask[:,:,0] = 0
    scores = get_average_score(mask,New_type)

    # tile through timesteps
    # [batch, seq_len, seq_len]
    past_labels = labels.unsqueeze(1).repeat(1, seq_len, 1)
    # masking padded position by padding_idx
    past_labels.masked_fill_((1 - mask).bool(), padding_idx)

    return past_labels, scores


def convert_to_future_labels(labels, new_times, src_tokens, New_type=None,padding_idx=1):
    """
    Args:
        padding_idx:
        labels: [batch, seq_len]

    Returns:
        future labels .
            [batch, seq_len, seq_len]
    """
    batch_size, seq_len = labels.size()
    _, src_len = src_tokens.size()
    seq_mask = src_tokens.ne(padding_idx)

    # use upper triangle to masking in descending manner
    # [batch, seq_len, seq_len]
    step_mask = torch.tril(labels.new_ones(seq_len, src_len), new_times-1).byte()
    mask = step_mask.unsqueeze(0) * seq_mask.unsqueeze(1)
    tmp_mask = labels.eq(padding_idx)
    mask = mask.masked_fill(tmp_mask.unsqueeze(-1), 0)
    scores = get_average_score(mask,New_type)

    # tile through timesteps
    # [batch, seq_len, seq_len]
    future_labels = src_tokens.unsqueeze(1).repeat(1, seq_len, 1)
    # masking padded position by padding_idx
    future_labels.masked_fill_((1 - mask).bool(), padding_idx)

    return future_labels, scores

class Criterion(nn.Module):
    """ Class for managing loss computation.

    """

    def _compute_loss(self, inputs, labels, **kwargs):
        """
        Compute the loss. Subclass must override this method.

        Args:
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        Returns:
            A non-reduced FloatTensor with shape (batch, )
        """
        raise NotImplementedError

    def forward(self, inputs, labels, normalization=1.0, reduce=True, **kwargs):
        """
        Compute loss given inputs and labels.

        Args:
            inputs: Input tensor of the criterion.
            labels: Label tensor of the criterion.
            reduce: Boolean value indicate whether the criterion should reduce the loss along the batch. If false,
                the criterion return a FloatTensor with shape (batch, ), otherwise a scalar.
            normalization: Normalization factor of the loss. Should be a float scalar or a FloatTensor with shape
                (batch, )
        """
        loss = self._compute_loss(inputs, labels, **kwargs).div(normalization)  # (batch, )

        if reduce:
            loss = loss.sum()

        return loss


class NMTCriterion(Criterion):
    """ A common used criterion for neural machine translation

    NMTCriterion is used for MLE training given golden target sample. Additional label_smoothing
    is supported.
    """

    def __init__(self, padding_idx=1, label_smoothing=0.0):

        super().__init__()

        self.padding_idx = padding_idx
        self.label_smoothing = label_smoothing

        if label_smoothing > 0:

            self.criterion = nn.KLDivLoss(size_average=False, reduce=False)

        else:
            self.criterion = nn.NLLLoss(size_average=False, ignore_index=padding_idx, reduce=False)

        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens, New_type):

        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.

        one_hot = torch.randn(1, num_tokens).type(New_type)
        one_hot.fill_(self.label_smoothing / (num_tokens - 2))
        one_hot[0][self.padding_idx] = 0

        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _compute_loss(self, inputs, labels, **kwargs):

        """
        Args:
            inputs (..., K): Expect logarithm probabilities.
            labels (...,): Index tensor. Should be the same size as inputs except the last dimension.
        """

        batch_size = labels.size(0)

        scores = self._bottle(inputs)  # [batch_size * seq_len, d_words]

        num_tokens = scores.size(-1)

        gtruth = labels.view(-1)

        if self.confidence < 1:
            # N: the number of samples
            # M: the number of labels
            tdata = gtruth.detach()

            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()  # mask of PAD

            one_hot = self._smooth_label(num_tokens)  # Do label smoothing
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)

            if mask.numel() > 0:
                tmp_.index_fill_(0, mask, 0)
            gtruth = tmp_.detach()

        loss = self.criterion(scores, gtruth).view((batch_size, -1)).sum(-1)

        return loss

class MultiTargetNMTCriterion(NMTCriterion):
    def __init__(self, padding_idx=1, label_smoothing=0.0):
        super().__init__(padding_idx,label_smoothing)
        
    def _construct_target(self, targets, target_scores, num_tokens):
        """
        Args:
            targets: A Tensor with shape [batch*length, max_target] represents the indices of
                targets in the vocabulary.
            target_scores: A Tensor with shape [batch*length, max_target] represents the
                probabilities of targets in the vocabulary.
            num_tokens: An Integer represents the total number of words.

        Returns:
            A Tensor with shape [batch*length, num_tokens].
        """
        # Initialize a temporary tensor.
        if self.confidence < 1:
            tmp = self._smooth_label(num_tokens, target_scores.dtype)  # Do label smoothing
            target_scores = target_scores * self.confidence
        else:
            tmp = torch.zeros(1, num_tokens)
        if targets.is_cuda:
            tmp = tmp.cuda()

        pad_positions = torch.nonzero(target_scores.sum(-1).eq(0)).squeeze()

        # [batch*length, num_tokens]
        tmp = tmp.repeat(targets.size(0), 1)

        if torch.numel(pad_positions) > 0:
            tmp.index_fill_(0, pad_positions, 0.)
        tmp.scatter_(1, targets, 0.)
        tmp.scatter_add_(1, targets, target_scores)

        return tmp

    def _compute_loss(self, inputs, labels, **kwargs):
        """
        Args:
            inputs: [batch, length, num_tokens]
            labels: [batch, length, max_target]
            **kwargs:

        Returns:
            A Tensor with shape of [batch,].
        """
        batch_size = labels.size(0)
        scores = self._bottle(inputs)  # [batch_size * seq_len, num_tokens]
        num_tokens = scores.size(-1)

        targets = self._bottle(labels)  # [batch_size * seq_len, max_target]
        target_scores = self._bottle(kwargs['target_scores'])

        # [batch*length, num_tokens]
        gtruth = self._construct_target(targets, target_scores, num_tokens).float()

        # [batch,]
        loss = self.criterion(scores, gtruth)
        loss = loss.sum(-1).view(batch_size, -1)  # [batch, seq_len]
        length_norm = kwargs['target_scores'].sum(-1).ne(0).float().sum(-1)  # [batch, ]
        loss = loss.sum(-1).div(length_norm)

        return loss