"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""

from os import sendfile
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable

NUM_ROUTING_ITERATIONS = 3
J = 4
DIMENSION = 256

class CapsuleLayer(nn.Module):
    def __init__(self, num_outcapsule, dim_in_caps, dim_out_caps, num_iterations, dim_context):
        super().__init__()

        self.num_iterations = num_iterations

        self.num_capsules = num_outcapsule
        self.dim_out_caps = dim_out_caps
        self.route_weights = nn.Parameter(torch.Tensor(self.num_capsules, dim_in_caps, dim_out_caps))

        self.linear_u_hat = nn.Linear(dim_out_caps, dim_out_caps, bias=False)
        self.linear_v = nn.Linear(dim_out_caps, dim_out_caps, bias=False)
        self.linear_c = nn.Linear(dim_context, dim_out_caps, bias=False)
        self.linear_delta = nn.Linear(dim_out_caps, 1, False)

        nn.init.xavier_uniform_(self.route_weights)
        nn.init.xavier_uniform_(self.linear_u_hat.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear_c.weight)
        nn.init.xavier_uniform_(self.linear_delta.weight)

    def squash(self, tensor, dim=-1):
        tensor = torch.squeeze(tensor, dim=-2) # batch * tgtlen * capsule * dim
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm)+1e-8)
    def capsule_project(self, input):
        batch_size, src_len, src_dim = input.size()
        input_tmp = input.contiguous().view(batch_size * src_len, -1)
        route_weight = self.route_weights.transpose(0, 1).reshape(src_dim, -1)
        priors_hat = input_tmp @ route_weight
        priors_hat = priors_hat.view(batch_size, src_len, self.num_capsules, -1)
        return priors_hat
    def compute_delta_sequence(self, priors, outputs, decoding_hid=None):
        """
        Args:
            priors: [batch_size, src_len, num_capsules, dim_out_caps]
            outputs: [batch_size, tgt_len, num_capsules, dim_out_caps]
            contexts: [batch_size, tgt_len, dim_context]

        Returns: Tensor. [batch_size, length, num_in_caps, num_out_caps]

        """
        # [batch_size, length, num_in_caps, num_out_caps, dim_out_caps]
        u = priors[:, None, :, :, :]
        v = outputs[:, :, None, :, :]
        # [batch_size, length, num_in_caps, num_out_caps, dim_out_caps]
        delta = self.linear_u_hat(u) + self.linear_v(v)
        # [batch, tgt_len, 1, 1, dim_context]
        c = decoding_hid[:, :, None, None, :]
        # [batch, length, num_in_caps, num_out_caps, dim_out_caps]
        delta = delta + self.linear_c(c)
        delta = torch.tanh(delta)
        # [batch_size, length, num_in_caps, num_out_caps]
        delta = self.linear_delta(delta).squeeze(-1)
        delta = torch.tanh(delta)
        return delta * (self.dim_out_caps ** -0.5)

    def forward(self, x, decoding_hid, new_times, encoder_mask=None):

        x = torch.transpose(x, 0, 1)
        src_len = x.size(1)
        tgt_len = decoding_hid.size(1)
        batch_size = x.size(0)
        capsule_num = self.num_capsules

        priors = self.capsule_project(input=x)  
        new_masked = x.new_zeros(tgt_len,src_len).masked_fill(mask = torch.tril(x.new_ones(src_len,tgt_len),diagonal=-new_times).transpose(0,1).bool(), value = torch.tensor(-np.inf,device=x.device))
        logits = x.new_zeros((batch_size, tgt_len, src_len, capsule_num))
        if encoder_mask is not None:
            encoder_mask = encoder_mask[:, None, :, None].expand_as(logits)
        

        for i in range(self.num_iterations):
            if encoder_mask is not None:
                logits = logits.masked_fill(encoder_mask, -1e18)
            probs = logits[:,None,:,:,:].transpose(-1,1).squeeze(-1) + new_masked
            probs = torch.exp(probs[:,:,:,:,None].transpose(-1,1).squeeze(1))
            probs = probs / (probs.sum(-1).unsqueeze(-1)+1e-8)

            outputs = self.squash((probs.unsqueeze(-1) * priors.unsqueeze(1)).sum(2)) 
            if i != self.num_iterations - 1:
                delta_logits = self.compute_delta_sequence(priors, outputs, decoding_hid)
                logits = logits + delta_logits

        return outputs  