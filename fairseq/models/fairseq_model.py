# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base classes for various fairseq models.
"""

from email import iterators
from typing import Dict, List, Optional
from xml.sax import default_parser_list

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from fairseq import utils
from fairseq.data import Dictionary
from fairseq.models import FairseqDecoder, FairseqEncoder
import numpy as np


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    nll_loss = -lprobs.gather(dim=-1, index=target.long())
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = ignore_index
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss

class BaseFairseqModel(nn.Module):
    """Base class for fairseq models."""

    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError('Model must implement the build_model method')

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, '')

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += '.'

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, 'upgrade_state_dict_named'):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, 'upgrade_state_dict'):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    def make_generation_fast_(self, **kwargs):
        """Optimize model for faster generation."""
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        seen = set()

        def apply_make_generation_fast_(module):
            if module != self and hasattr(module, 'make_generation_fast_') \
                    and module not in seen:
                seen.add(module)
                module.make_generation_fast_(**kwargs)

        self.apply(apply_make_generation_fast_)

        def train(mode=True):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')
        # this model should no longer be used for training
        self.eval()
        self.train = train

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""
        seen = set()

        def apply_prepare_for_onnx_export_(module):
            if module != self and hasattr(module, 'prepare_for_onnx_export_') \
                    and module not in seen:
                seen.add(module)
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', **kwargs):
        """
        Load a :class:`~fairseq.models.FairseqModel` from a pre-trained model
        file. Downloads and caches the pre-trained model file if needed.

        The base implementation returns a
        :class:`~fairseq.hub_utils.GeneratorHubInterface`, which can be used to
        generate translations or sample from language models. The underlying
        :class:`~fairseq.models.FairseqModel` can be accessed via the
        *generator.models* attribute.

        Other models may override this to implement custom hub interfaces.

        Args:
            model_name_or_path (str): either the name of a pre-trained model to
                load or a path/URL to a pre-trained model state dict
            checkpoint_file (str, optional): colon-separated list of checkpoint
                files in the model archive to ensemble (default: 'model.pt')
            data_name_or_path (str, optional): point args.data to the archive
                at the given path/URL. Can start with '.' or './' to reuse the
                model archive path.
        """
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            **kwargs,
        )
        print(x['args'])
        return hub_utils.GeneratorHubInterface(x['args'], x['task'], x['models'])

    @classmethod
    def hub_models(cls):
        return {}

def Linear(in_features, out_features, bias=False, dtype=None):
    if dtype is not None:
        m = nn.Linear(in_features, out_features, bias, dtype=dtype)
    else:
        m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def Embedding(num_embeddings, embedding_dim, padding_idx, weight):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, _weight=weight)
    return m

class FairseqEncoderDecoderModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        if self.decoder.classifier_training:
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)

            self.fc = Linear(in_features=self.decoder.embed_tokens.embedding_dim*2, out_features=1024)
            self.out = Linear(in_features=1024, out_features=1)
            # project the tokens and actions into the same space
            self.emd_fc = Linear(in_features=self.decoder.embed_tokens.embedding_dim*2, out_features=512)
            self.act_fc = Linear(in_features=self.decoder.embed_tokens.embedding_dim*1, out_features=512)
            # project the features to the action space
            self.true_out = Linear(in_features=512, out_features=2)
            # lSTM layer parameter
            self.lstm_layer=torch.nn.LSTM(input_size=512*2, hidden_size=512, num_layers=1,
                                    bias=True, batch_first=False, bidirectional=False)
            self.lstm_layer.reset_parameters()
            # action embedding
            self.action_embedding = torch.nn.Embedding(2, 512)
            self.action_embedding.reset_parameters()
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, tgt_tokens, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        if self.decoder.multipath_training:
            encoder_out, src_embedding = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
            new_times = torch.randint(1,src_tokens.size(1)+1,(1,)).cuda(device=src_tokens.device).item()
            decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, tgt_tokens=tgt_tokens, src_tokens=src_tokens,new_times=new_times, **kwargs)
            return decoder_out, None
        # get bsz, srclen, tgtlen
        bsz = src_tokens.size(0)
        tgtLen = tgt_tokens.size(-1)
        src_mask = src_tokens.eq(self.encoder.padding_idx)
        src_lengths = src_mask.size(-1) - torch.sum(src_mask, dim=-1)
        tgt_mask = tgt_tokens.eq(self.encoder.padding_idx)
        tgt_lengths = tgt_mask.size(-1) - torch.sum(tgt_mask, dim=-1)
        predLoss = None

        # get encoder hidden states
        encoder_out, src_embedding = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # generate binary search for each target token
        lbound = (torch.arange(tgtLen).cuda()).expand(bsz, tgtLen).float() * (src_lengths/tgt_lengths).unsqueeze(-1) + self.decoder.left_bound
        rbound = (torch.arange(tgtLen).cuda()).expand(bsz, tgtLen).float() * (src_lengths/tgt_lengths).unsqueeze(-1) + self.decoder.right_bound
        rbound[rbound > src_lengths.unsqueeze(-1)] = src_lengths.float().unsqueeze(-1).expand(bsz, tgtLen)[rbound > src_lengths.unsqueeze(-1)]
        lbound[lbound > src_lengths.unsqueeze(-1)] = src_lengths.float().unsqueeze(-1).expand(bsz, tgtLen)[lbound > src_lengths.unsqueeze(-1)]
        rbound = rbound.int()
        lbound = lbound.int()
        
        # generating probability for each target token
        lProb = None
        rProb = None
        mProb = None

        # get the translation probability of target tokens for the left bound
        lmask = self.generMask(lbound, tgtLen, src_mask.size(-1))
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, fc=None, out=None, new_times=lmask, **kwargs)
        lProb, _ = self.getProb(decoder_out[0], tgt_tokens)

        # get the translation probability of target tokens for the right bound
        lmask = self.generMask(rbound, tgtLen, src_mask.size(-1))
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, fc=None, out=None, new_times=lmask, **kwargs)
        rProb, rinde = self.getProb(decoder_out[0], tgt_tokens)

        # binary search for each target token to find the best path
        while True:
            mbound = ((lbound + rbound) / 2).int()

            lmask = self.generMask(mbound, tgtLen, src_mask.size(-1))
            decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, fc=None, out=None, new_times=lmask, **kwargs)
            mProb, minde = self.getProb(decoder_out[0], tgt_tokens)

            if not self.isStop(lbound, rbound, tgt_mask):
                break
            
            # choose the left interval or the right interval
            midMask = (((lProb >= rProb) | ((lProb + rProb) <= (2 * mProb))) & (rbound - lbound > 1)) | ((rbound - lbound ==1) & (minde==rinde))
            lbound = lbound.masked_fill((1 - midMask.int()).bool(), 0) + mbound.masked_fill(midMask, 0) + ((rbound - lbound==1) & ((1-midMask.int()).bool())).int()
            rbound = rbound.masked_fill(midMask, 0) + mbound.masked_fill((1 - midMask.int()).bool(), 0)

            # modify the probability of the left interval
            rProb = rProb.masked_fill(midMask, 0) + mProb.masked_fill((1 - midMask.int()).bool(), 0)
            rinde = rinde.masked_fill(midMask, 0) + minde.masked_fill((1 - midMask.int()).bool(), 0)
            lProb = lProb.masked_fill((1 - midMask.int()).bool(), 0) + mProb.masked_fill(midMask, 0)
        
        # generate the tokens embeddings
        if self.decoder.classifier_training:
            dec_hidd = self.generate_translation_embed(decoder_out, prev_output_tokens)
            predLoss = self.getPrecitLoss(enc_hidd=src_embedding, dec_hidd=dec_hidd, division=mbound, src_lengths=src_lengths.detach(), tgt_mask=tgt_mask.detach(), tgtLen=tgt_lengths)
        return decoder_out, predLoss

    def generate_translation_embed(self, decoder_out, prev_output_tokens):
        tgt_prob = decoder_out[0].detach()
        index = torch.max(tgt_prob, dim=-1)[1]
        index = torch.cat((prev_output_tokens[:, :1], index[:, 1:]), dim=-1)
        dec_hidd = self.decoder.embed_scale * self.decoder.embed_tokens(index)

        return dec_hidd.detach()

    # convert the the number of read tokens into the action sequence, where 0 means read and 1 means write
    def generate_action(self, division, src_len, tgt_mask):
        bsz = division.size(0)
        tgt_len = division.size(1)
        division = division.masked_fill(tgt_mask, src_len)
        action_seq = torch.zeros(bsz, tgt_len+src_len).cuda().int()
        tempor = torch.arange(tgt_len).cuda().unsqueeze(0)
        action_seq = action_seq.scatter(-1, (division + tempor).long(), 1)

        return action_seq

    def getPrecitLoss(self, enc_hidd, dec_hidd, division, src_lengths, tgt_mask, tgtLen):
        
        bsz = dec_hidd.size(0)
        src_len = enc_hidd.size(1)
        tgt_len = dec_hidd.size(1)
        hidd_dim = dec_hidd.size(-1)
        division = self.norm(division, tgt_len)
        # get the action sequence
        action_seq = self.generate_action(division, src_len, tgt_mask)

        # expand the hidden states to match the action sequence
        tgt_index = torch.cumsum(action_seq, dim=-1) - action_seq
        tgt_index[tgt_index > (tgt_len - 1)] = tgt_len - 1
        dec_hidd = torch.gather(dec_hidd, 1, tgt_index.long().unsqueeze(-1).expand(bsz, tgt_len+src_len, hidd_dim))
        dec_hidd = dec_hidd[:, 1:, :]
        src_index = (torch.cumsum(1 - action_seq, dim=-1) - (1 - action_seq) - 1)[:, 1:]
        enc_hidd = torch.gather(enc_hidd, 1, src_index.long().unsqueeze(-1).expand(bsz, tgt_len+src_len-1, hidd_dim))

        action_embedding = self.action_embedding(action_seq[:, :-1].long())

        # input the hidden states and action sequence into the LSTM layer
        midd_fea = self.emd_fc(torch.cat((enc_hidd, dec_hidd), dim=-1))
        midd_fea = F.relu(midd_fea)
        action_embedding = F.relu(self.act_fc(action_embedding))
        midd_fea = torch.cat((midd_fea, action_embedding), dim=-1)
        midd_fea, (_, _) = self.lstm_layer(midd_fea.transpose(0, 1))
        midd_fea = midd_fea.transpose(0, 1)
        midd_fea = self.true_out(midd_fea)

        # generate the mask for the action loss function
        mask = torch.zeros(bsz, src_len+tgt_len-1).cuda()
        mask = mask.scatter(-1, (src_lengths + tgtLen).long().unsqueeze(-1) - 2, 1)
        mask = torch.flip(torch.cumsum(torch.flip(mask, dims=[-1]), -1), dims=[-1]).int()
        mask = mask.bool()
        action_seq = action_seq[:, 1:]

        eps_i = self.decoder.action_loss_smoothing

        loss = self.compute_loss(action_seq, midd_fea.unsqueeze(0), eps_i, mask, reduce=True)

        return [loss, float(torch.sum(mask.int()))]

    def compute_loss(self, target, net_output, eps, mask, reduce=True):
        lprobs = self.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        loss = label_smoothed_nll_loss(
            lprobs, target, eps, ignore_index=mask, reduce=reduce,
        )

        return loss

    # get the translation probability of ground-truth tokens
    def getProb(self, prob, inde):
        prob = F.softmax(prob, dim=-1)
        tgt_prob = torch.gather(prob, dim=-1, index=inde.unsqueeze(-1))
        _, inde = torch.max(prob, dim=-1)
        return tgt_prob.squeeze(-1), inde

    # prevent the source to be leaked in the previous steps
    def norm(self, mbound, tgtLen):
        for i in range(tgtLen - 1):
            tmp_prob = mbound[:, i] > mbound[:, i+1]
            mbound[tmp_prob,i+1] = mbound[tmp_prob,i]
        return mbound

    def localMaskNorm(self, mask, tgtLen):
        if mask.size(-1) >= 3:
            mask[:, -1] = mask[:, -1] * False
            for i in range(tgtLen-2, 0, -1):
                tmp_mask = mask[:, i-1] == mask[:, i+1]
                mask[tmp_mask, i] = mask[tmp_mask, i+1]
            mask[:, 0] = (mask[:, 0].int() + mask[:, 1].int() + mask[:, 2].int()) >= 2
        return mask

    # generate the mask for cross attention
    def generMask(self, mbound, tgt_len, src_len):
        can_mask = torch.zeros(mbound.size(0), tgt_len, src_len).cuda()
        can_mask = can_mask.scatter(-1, mbound.long().unsqueeze(-1) - 1, 1)
        can_mask = torch.flip(torch.cumsum(torch.flip(can_mask, dims=[-1]), -1), dims=[-1])
        maskk = (1 - can_mask).bool()
        new_times = (1 - can_mask).masked_fill(maskk, -np.inf)
        new_times = new_times.unsqueeze(1).expand(new_times.size(0), self.decoder.layers[0].encoder_attn.num_heads, new_times.size(1), new_times.size(2))
        new_times = new_times.contiguous().view(-1, new_times.size(-2), new_times.size(-1))
        return new_times

    # check whether the binary search is over
    def isStop(self, lbound, rbound, mask):
        res = lbound < rbound
        res = res.masked_fill(mask, False)
        return torch.sum(res) > 0


    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


class FairseqModel(FairseqEncoderDecoderModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        utils.deprecation_warning(
            'FairseqModel is deprecated, please use FairseqEncoderDecoderModel '
            'or BaseFairseqModel instead',
            stacklevel=4,
        )


class FairseqMultiModel(BaseFairseqModel):
    """Base class for combining multiple encoder-decoder models."""

    def __init__(self, encoders, decoders):
        super().__init__()
        assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            assert isinstance(encoders[key], FairseqEncoder)
            assert isinstance(decoders[key], FairseqDecoder)

        self.models = nn.ModuleDict({
            key: FairseqModel(encoders[key], decoders[key])
            for key in self.keys
        })

    @staticmethod
    def build_shared_embeddings(
        dicts: Dict[str, Dictionary],
        langs: List[str],
        embed_dim: int,
        build_embedding: callable,
        pretrained_embed_path: Optional[str] = None,
    ):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                '--share-*-embeddings requires a joined dictionary: '
                '--share-encoder-embeddings requires a joined source '
                'dictionary, --share-decoder-embeddings requires a joined '
                'target dictionary, and --share-all-embeddings requires a '
                'joint source + target dictionary.'
            )
        return build_embedding(
            shared_dict, embed_dim, pretrained_embed_path
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        decoder_outs = {}
        for key in self.keys:
            encoder_out = self.models[key].encoder(src_tokens, src_lengths, **kwargs)
            decoder_outs[key] = self.models[key].decoder(
                prev_output_tokens, encoder_out, **kwargs,
            )
        return decoder_outs

    def max_positions(self):
        """Maximum length supported by the model."""
        return {
            key: (self.models[key].encoder.max_positions(), self.models[key].decoder.max_positions())
            for key in self.keys
        }

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return min(model.decoder.max_positions() for model in self.models.values())

    @property
    def encoder(self):
        return self.models[self.keys[0]].encoder

    @property
    def decoder(self):
        return self.models[self.keys[0]].decoder


class FairseqLanguageModel(BaseFairseqModel):
    """Base class for decoder-only models.

    Args:
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, **kwargs):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, seq_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        return self.decoder(src_tokens, **kwargs)

    def extract_features(self, src_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, seq_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        return self.decoder.extract_features(src_tokens, **kwargs)

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.decoder.max_positions()

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()

    @property
    def supported_targets(self):
        return {'future'}


class FairseqEncoderModel(BaseFairseqModel):
    """Base class for encoder-only models.

    Args:
        encoder (FairseqEncoder): the encoder
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        assert isinstance(self.encoder, FairseqEncoder)

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        Run the forward pass for a encoder-only model.

        Feeds a batch of tokens through the encoder to generate features.

        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the encoder's output, typically of shape `(batch, src_len, features)`
        """
        return self.encoder(src_tokens, src_lengths, **kwargs)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output['encoder_out']
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions()
