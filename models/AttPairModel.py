# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import models
import cPickle

from .CaptionModel import CaptionModel

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class AttPairModel(CaptionModel):
    def __init__(self, opt):
        super(AttPairModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
            self.logit2 = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
            self.logit2 = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in
                          range(opt.logit_layers - 1)]
            self.logit2 = nn.Sequential(
                *(reduce(lambda x, y: x + y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ctx2att2 = nn.Linear(self.rnn_size, self.att_hid_size)
        self.prev_sent_emb = nn.Linear(self.rnn_size, self.att_feat_size)
        self.prev_sent_wrap = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.rank_model = None

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _prepare_feature2(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att2(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None, masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq1 = seq[:, 0, :]
        seq2 = seq[:, 1, :]

        outputs1 = fc_feats.new_zeros(batch_size, seq1.size(1) - 1, self.vocab_size + 1)
        outputs2 = fc_feats.new_zeros(batch_size, seq2.size(1) - 1, self.vocab_size + 1)
        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        hiddens = []

        for i in range(seq1.size(1) - 1):
            it = seq1[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq1[:, i].sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks)
            hiddens.append(state[0])
            output = F.log_softmax(self.logit(output), dim=1)
            outputs1[:, i] = output

        hiddens = self.prev_sent_emb(torch.cat(hiddens, 0).transpose(0,1).contiguous())
        prev_hidden_masks = masks[:, :hiddens.size(1)]
        p_hiddens = pack_wrapper(self.prev_sent_wrap, hiddens, prev_hidden_masks)
        # clip hidden states and masks
        hiddens = hiddens[:, :p_hiddens.size(1), :]
        prev_hidden_masks = prev_hidden_masks[:, :p_hiddens.size(1)]

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature2(fc_feats, att_feats, att_masks)
        state = self.init_hidden(batch_size)
        for i in range(seq2.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq2[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq2[:, i].data.clone()
                    prob_prev = torch.exp(outputs2[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq2[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq2[:, i].sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core2(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks, hiddens, p_hiddens, prev_hidden_masks)
            output = F.log_softmax(self.logit2(output), dim=1)
            outputs2[:, i] = output

        return outputs1, outputs2

    def forward_greedy(self, fc_feats, att_feats, seq, att_masks=None, masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)
        seq2 = seq[:, 1, :]

        outputs2 = fc_feats.new_zeros(batch_size, seq2.size(1) - 1, self.vocab_size + 1)
        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost
        seq1 = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs1 = fc_feats.new_zeros(batch_size, self.seq_length)
        hiddens = []
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            xt = self.embed(it)

            output, state = self.core(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks)
            hiddens.append(state[0])
            logprobs = F.log_softmax(self.logit(output), dim=1)
            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            sampleLogprobs, it = torch.max(logprobs.data, 1)
            it = it.view(-1).long()
            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq1[:, t] = it
            seqLogprobs1[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        hiddens = self.prev_sent_emb(torch.cat(hiddens, 0).transpose(0, 1).contiguous())
        prev_hidden_masks = hiddens.new(batch_size, hiddens.size(1)).zero_()
        for i in range(seq1.size(0)):
            prev_hidden_masks[i, 0] = 1
            for j in range(seq1.size(1)):
                if seq1[i, j] != 0:
                    prev_hidden_masks[i, j] = 1
                else:
                    break

        p_hiddens = pack_wrapper(self.prev_sent_wrap, hiddens, prev_hidden_masks)
        hiddens = hiddens[:, :p_hiddens.size(1), :]
        prev_hidden_masks = prev_hidden_masks[:, :p_hiddens.size(1)]
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature2(fc_feats, att_feats, att_masks)

        state = self.init_hidden(batch_size)
        for i in range(seq2.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq2[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq2[:, i].data.clone()
                    prob_prev = torch.exp(outputs2[:, i - 1].detach())  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq2[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq2[:, i].sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core2(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks, hiddens,
                                       p_hiddens, prev_hidden_masks)
            output = F.log_softmax(self.logit2(output), dim=1)
            outputs2[:, i] = output

        return outputs2

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, hiddens=None, p_hiddens=None, prev_hidden_masks=None, state=None):
        # 'it' contains a word index
        xt = self.embed(it)
        if hiddens is None:
            output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            output, state = self.core2(xt, fc_feats, att_feats, p_att_feats, state, att_masks, hiddens,
                                   p_hiddens, prev_hidden_masks)
            logprobs = F.log_softmax(self.logit2(output), dim=1)

        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        two_step_beam = opt.get('two_step_beam', 0)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq1 = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs1 = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(*((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(*((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, None, None, None, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, None, None, None, opt=opt)
            seq1[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs1[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods

        # forward to get hidden states
        if not two_step_beam:
            state = self.init_hidden(batch_size)
            hiddens = []
            # feed <bos>
            it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            xt = self.embed(it)
            output, state = self.core(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks)
            seq1 = seq1.transpose(0, 1)
            if torch.cuda.is_available():
                seq1 = seq1.cuda()
            for i in range(seq1.size(1) - 1):
                it = seq1[:, i].clone()
                # break if all the sequences end
                if i >= 1 and seq1[:, i].sum() == 0:
                    break
                xt = self.embed(it)

                output, state = self.core(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks)
                hiddens.append(state[0])
                #output = F.log_softmax(self.logit(output), dim=1)
            hiddens = self.prev_sent_emb(torch.cat(hiddens, 0).transpose(0, 1).contiguous())
            prev_hidden_masks = hiddens.new(hiddens.size(0), max(hiddens.size(1), seq1.size(1))).zero_()
            for i in range(seq1.size(0)):
                for j in range(seq1.size(1)):
                    if seq1[i, j] != 0:
                        prev_hidden_masks[i, j] = 1
                    else:
                        break
            prev_hidden_masks[:, 0] = 1  ## seq_len > 0
            prev_hidden_masks = torch.cat([prev_hidden_masks.new(batch_size, 1).zero_() + 1, prev_hidden_masks],
                                          1).contiguous()[:, :hiddens.size(1)]

        else:
            seq1 = seq1.transpose(0, 1)
            all_beam_seqs = []
            for i in range(batch_size):
                seqs = torch.cat([_['seq'].unsqueeze(0) for _ in self.done_beams[i]], 0)
                all_beam_seqs.append(seqs)
            all_beam_seqs = torch.cat(all_beam_seqs, 0).contiguous()

            state = self.init_hidden(batch_size * beam_size)
            hiddens = []
            # feed <bos>
            it = fc_feats.new_zeros(batch_size * beam_size, dtype=torch.long)
            xt = self.embed(it)

            # batch * size -> (batch * beam_size) * size
            new_p_fc_feats = p_fc_feats.unsqueeze(1).expand(batch_size, beam_size, p_fc_feats.size(1)).reshape(batch_size * beam_size, p_fc_feats.size(1))
            new_p_att_feats = p_att_feats.unsqueeze(1).expand(batch_size, beam_size, p_att_feats.size(1), p_att_feats.size(2)).\
                reshape(batch_size * beam_size, p_att_feats.size(1), p_att_feats.size(2))
            new_pp_att_feats = pp_att_feats.unsqueeze(1).expand(batch_size, beam_size, pp_att_feats.size(1),
                             pp_att_feats.size(2)).reshape(batch_size * beam_size, pp_att_feats.size(1), pp_att_feats.size(2))
            new_p_att_masks = p_att_masks.unsqueeze(1).expand(batch_size, beam_size, p_att_masks.size(1)).reshape(batch_size * beam_size, p_att_masks.size(1))

            output, state = self.core(xt, new_p_fc_feats, new_p_att_feats, new_pp_att_feats, state, new_p_att_masks)

            if torch.cuda.is_available():
                all_beam_seqs = all_beam_seqs.cuda()
            for i in range(all_beam_seqs.size(1) - 1):
                it = all_beam_seqs[:, i].clone()
                # break if all the sequences end
                if i >= 1 and all_beam_seqs[:, i].sum() == 0:
                    break
                xt = self.embed(it)

                output, state = self.core(xt, new_p_fc_feats, new_p_att_feats, new_pp_att_feats, state, new_p_att_masks)
                hiddens.append(state[0])
            hiddens = self.prev_sent_emb(torch.cat(hiddens, 0).transpose(0, 1).contiguous())
            prev_hidden_masks = hiddens.new(hiddens.size(0), hiddens.size(1)).zero_()
            for i in range(hiddens.size(0)):
                for j in range(hiddens.size(1)):
                    if all_beam_seqs[i, j] != 0:
                        prev_hidden_masks[i, j] = 1
                    else:
                        break
            prev_hidden_masks[:, 0] = 1  ## seq_len > 0
            prev_hidden_masks = torch.cat([prev_hidden_masks.new(prev_hidden_masks.size(0), 1).zero_() + 1, prev_hidden_masks],
                                          1).contiguous()[:, :hiddens.size(1)]

        p_hiddens = pack_wrapper(self.prev_sent_wrap, hiddens, prev_hidden_masks)
        hiddens = hiddens[:, :p_hiddens.size(1), :]
        prev_hidden_masks = prev_hidden_masks[:, :p_hiddens.size(1)]


        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature2(fc_feats, att_feats, att_masks)
        state = self.init_hidden(batch_size)
        seq2 = fc_feats.new_zeros((self.seq_length, batch_size), dtype=torch.long)
        seqLogprobs2 = fc_feats.new_zeros(self.seq_length, batch_size)

        self.done_beams = [[] for _ in range(batch_size)]
        if two_step_beam:
            for k in range(batch_size):
                state = self.init_hidden(beam_size ** 2)
                tmp_fc_feats = p_fc_feats[k:k + 1].expand(beam_size ** 2, p_fc_feats.size(1))
                tmp_att_feats = p_att_feats[k:k + 1].expand(*((beam_size ** 2,) + p_att_feats.size()[1:])).contiguous()
                tmp_p_att_feats = pp_att_feats[k:k + 1].expand(*((beam_size ** 2,) + pp_att_feats.size()[1:])).contiguous()
                tmp_att_masks = p_att_masks[k:k + 1].expand(
                    *((beam_size ** 2,) + p_att_masks.size()[1:])).contiguous() if att_masks is not None else None
                tmp_hiddens = hiddens[k * beam_size: (k + 1) * beam_size].unsqueeze(0).expand(beam_size,beam_size, hiddens.size(1), hiddens.size(2)).\
                    reshape(beam_size ** 2, hiddens.size(1), hiddens.size(2)).contiguous()
                tmp_p_hiddens = p_hiddens[k * beam_size: (k + 1) * beam_size].unsqueeze(0).expand(beam_size,beam_size, p_hiddens.size(1), p_hiddens.size(2)).\
                    reshape(beam_size ** 2, p_hiddens.size(1), p_hiddens.size(2)).contiguous()
                tmp_prev_hidden_masks = prev_hidden_masks[k * beam_size: (k + 1) * beam_size].unsqueeze(0).expand(beam_size, beam_size, prev_hidden_masks.size(1)).\
                    reshape(beam_size ** 2, prev_hidden_masks.size(1)).contiguous()

                for t in range(1):
                    if t == 0:  # input <bos>
                        it = fc_feats.new_zeros([beam_size ** 2], dtype=torch.long)
                    logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                              tmp_att_masks, tmp_hiddens, tmp_p_hiddens,
                                                              tmp_prev_hidden_masks, state)
                opt['beam_size'] = beam_size ** 2
                self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                      tmp_att_masks, tmp_hiddens,
                                                      tmp_p_hiddens, tmp_prev_hidden_masks, opt=opt)
                seq2[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs2[:, k] = self.done_beams[k][0]['logps']
        else:
            for k in range(batch_size):
                state = self.init_hidden(beam_size)
                tmp_fc_feats = p_fc_feats[k:k + 1].expand(beam_size, p_fc_feats.size(1))
                tmp_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()
                tmp_p_att_feats = pp_att_feats[k:k + 1].expand(*((beam_size,) + pp_att_feats.size()[1:])).contiguous()
                tmp_att_masks = p_att_masks[k:k + 1].expand(
                    *((beam_size,) + p_att_masks.size()[1:])).contiguous() if att_masks is not None else None
                tmp_hiddens = hiddens[k: k + 1].expand(*((beam_size,) + hiddens.size()[1:])).contiguous()
                tmp_p_hiddens = p_hiddens[k: k + 1].expand(*((beam_size,) + p_hiddens.size()[1:])).contiguous()
                tmp_prev_hidden_masks = prev_hidden_masks[k: k + 1].expand(*((beam_size,) + prev_hidden_masks.size()[1:])).contiguous()



                for t in range(1):
                    if t == 0:  # input <bos>
                        it = fc_feats.new_zeros([beam_size], dtype=torch.long)
                    logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                              tmp_att_masks, tmp_hiddens, tmp_p_hiddens, tmp_prev_hidden_masks, state)

                self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                      tmp_att_masks, tmp_hiddens,
                                               tmp_p_hiddens, tmp_prev_hidden_masks, opt=opt)
                seq2[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs2[:, k] = self.done_beams[k][0]['logps']

        return seq1.cpu(), seq2.transpose(0, 1)#, seqLogprobs2.transpose(0, 1)

    def beam_sample(self, fc_feats, att_feats, att_masks, opt={}):
        # (1) Run the encoder on the src.
        eval_ = False
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        ind = range(batch_size)
        decState = self.init_hidden(beam_size * batch_size)
        tmp_fc_feats = p_fc_feats.repeat(beam_size, 1)
        tmp_att_feats = p_att_feats.repeat(beam_size, 1, 1)
        tmp_p_att_feats = pp_att_feats.repeat(beam_size, 1, 1)
        tmp_att_masks = p_att_masks.repeat(beam_size, 1) if att_masks is not None else None

        #  (1b) Initialize for the decoder.
        def var(a):
            return torch.tensor(a, requires_grad=False)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        # contexts = rvar(contexts.data)
        # contexts = rvar(contexts)

        # if self.config.cell == 'lstm':
        #     decState = (rvar(encState[0]), rvar(encState[1]))
        # else:
        #     decState = rvar(encState)

        beam = [models.Beam(beam_size, n_best=1,
                            cuda=1, length_norm=0)
                for __ in range(batch_size)]
        # if self.decoder.attention is not None:
        #     self.decoder.attention.init_context(contexts)

        # (2) run the decoder to generate sentences, using beam search.

        for i in range(self.seq_length):
            if all((b.done() for b in beam)):
                break
            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))
            # Run one step.
            output, decState = self.get_logprobs_state(inp, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, None, None, None, decState)

            #self.get_logprobs_state(inp, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, decState)
            # decOut: beam x rnn_size
            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(output)
            # UNK
            output[:, :, output.size(2) - 1] = output[:, :, output.size(2) - 1] - 1000
            # beam x tgt_vocab
            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output[:, j])
                b.beam_update(decState, j)
        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []
        if eval_:
            allWeight = []

        # for j in ind.data:
        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            if eval_:
                weight = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                if eval_:
                    weight.append(att)
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            if eval_:
                allWeight.append(weight[0])
        seq1 = torch.zeros(batch_size, self.seq_length).long()
        for i in range(batch_size):
            for j, w in enumerate(allHyps[i]):
                seq1[i, j] = w
        tmp_seq1 = seq1.clone()
        # forward to get hidden_states
        state = self.init_hidden(batch_size)
        hiddens = []
        # feed <bos>
        it = fc_feats.new_zeros(batch_size, dtype=torch.long)
        xt = self.embed(it)
        output, state = self.core(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks)

        if torch.cuda.is_available():
            seq1 = seq1.cuda()
        for i in range(seq1.size(1) - 1):
            it = seq1[:, i].clone()
            if i >= 1 and seq1[:, i].sum() == 0:
                break
            xt = self.embed(it)
            output, state = self.core(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks)
            hiddens.append(state[0])
        hiddens = self.prev_sent_emb(torch.cat(hiddens, 0).transpose(0, 1).contiguous())
        prev_hidden_masks = hiddens.new(hiddens.size(0), max(hiddens.size(1), seq1.size(1))).zero_()
        for i in range(seq1.size(0)):
            for j in range(seq1.size(1)):
                if seq1[i, j] != 0:
                    prev_hidden_masks[i, j] = 1
                else:
                    break
        prev_hidden_masks[:, 0] = 1  ## seq_len > 0
        prev_hidden_masks = torch.cat([prev_hidden_masks.new(batch_size, 1).zero_() + 1, prev_hidden_masks],
                                      1).contiguous()[:, :hiddens.size(1)]
        p_hiddens = pack_wrapper(self.prev_sent_wrap, hiddens, prev_hidden_masks)
        hiddens = hiddens[:, :p_hiddens.size(1), :]
        prev_hidden_masks = prev_hidden_masks[:, :p_hiddens.size(1)]
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature2(fc_feats, att_feats, att_masks)
        ind = range(batch_size)
        decState = self.init_hidden(beam_size * batch_size)
        tmp_fc_feats = p_fc_feats.repeat(beam_size, 1)
        tmp_att_feats = p_att_feats.repeat(beam_size, 1, 1)
        tmp_p_att_feats = pp_att_feats.repeat(beam_size, 1, 1)
        tmp_att_masks = p_att_masks.repeat(beam_size, 1) if att_masks is not None else None
        tmp_hiddens = hiddens.repeat(beam_size, 1, 1)
        tmp_p_hiddens = p_hiddens.repeat(beam_size, 1, 1)
        tmp_prev_hidden_masks = prev_hidden_masks.repeat(beam_size, 1)
        # SECOND DECODING
        beam = [models.Beam(beam_size, n_best=1,
                            cuda=1, length_norm=0)
                for __ in range(batch_size)]
        # if self.decoder.attention is not None:
        #     self.decoder.attention.init_context(contexts)
        # (2) run the decoder to generate sentences, using beam search.

        for i in range(self.seq_length):
            if all((b.done() for b in beam)):
                break
            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))

            # Run one step.
            output, decState = self.get_logprobs_state(inp, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                       tmp_att_masks, tmp_hiddens, tmp_p_hiddens, tmp_prev_hidden_masks, decState)

            # decOut: beam x rnn_size
            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(output)
            # UNK
            output[:, :, output.size(2) - 1] = output[:, :, output.size(2) - 1] - 1000
            # beam x tgt_vocab
            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output[:, j])
                b.beam_update(decState, j)
        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []
        if eval_:
            allWeight = []

        # for j in ind.data:
        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            if eval_:
                weight = []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                if eval_:
                    weight.append(att)
            # print(hyps, scores, ks)
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            # allAttn.append(attn[0])
            if eval_:
                allWeight.append(weight[0])

        if eval_:
            return allHyps, allAttn, allWeight
        seq2 = torch.zeros(batch_size, self.seq_length).long()
        for i in range(batch_size):
            for j, w in enumerate(allHyps[i]):

                seq2[i, j] = w

        return tmp_seq1, seq2

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}, return_prob=False):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            #return self._sample_beam(fc_feats, att_feats, att_masks, opt)
            return self.beam_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq1 = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs1 = fc_feats.new_zeros(batch_size, self.seq_length)
        hiddens = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            xt = self.embed(it)

            output, state = self.core(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks)
            hiddens.append(state[0])
            logprobs = F.log_softmax(self.logit(output), dim=1)
            #logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq1[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq1[:, t] = it
            seqLogprobs1[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        hiddens = self.prev_sent_emb(torch.cat(hiddens, 0).transpose(0, 1).contiguous())
        prev_hidden_masks = hiddens.new(batch_size, hiddens.size(1)).zero_()
        for i in range(seq1.size(0)):
            for j in range(seq1.size(1)):
                if seq1[i, j] != 0:
                    prev_hidden_masks[i, j] = 1
                else:
                    break
        prev_hidden_masks[:, 0] = 1 ## seq_len > 0
        prev_hidden_masks = torch.cat([prev_hidden_masks.new(batch_size, 1).zero_() + 1, prev_hidden_masks], 1).contiguous()[:, :hiddens.size(1)]

        p_hiddens = pack_wrapper(self.prev_sent_wrap, hiddens, prev_hidden_masks)
        hiddens = hiddens[:, :p_hiddens.size(1), :]
        prev_hidden_masks = prev_hidden_masks[:, :p_hiddens.size(1)]
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature2(fc_feats, att_feats, att_masks)
        state = self.init_hidden(batch_size)
        seq2 = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs2 = fc_feats.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            xt = self.embed(it)

            output, state = self.core2(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks, hiddens, p_hiddens, prev_hidden_masks)
            logprobs = F.log_softmax(self.logit2(output), dim=1)
            #logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq1[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq2[:, t] = it
            seqLogprobs2[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        if return_prob:
            return seq2, seqLogprobs2
        return seq1, seq2

    def sample_first(self, fc_feats, att_feats, att_masks=None, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq1 = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs1 = fc_feats.new_zeros(batch_size, self.seq_length)
        hiddens = []
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            xt = self.embed(it)

            output, state = self.core(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks)
            hiddens.append(state[0])
            logprobs = F.log_softmax(self.logit(output), dim=1)
            # logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq1[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq1[:, t] = it
            seqLogprobs1[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq1, seqLogprobs1

    def sample_second(self, fc_feats, att_feats, seq, masks, att_masks=None, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        seq1 = seq[:, 0, :]

        hiddens = []

        for i in range(seq1.size(1) - 1):
            it = seq1[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq1[:, i].sum() == 0:
                break

            xt = self.embed(it)
            output, state = self.core(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks)
            hiddens.append(state[0])

        hiddens = self.prev_sent_emb(torch.cat(hiddens, 0).transpose(0, 1).contiguous())
        prev_hidden_masks = masks[:, :hiddens.size(1)]
        p_hiddens = pack_wrapper(self.prev_sent_wrap, hiddens, prev_hidden_masks)
        # clip hidden states and masks
        hiddens = hiddens[:, :p_hiddens.size(1), :]
        prev_hidden_masks = prev_hidden_masks[:, :p_hiddens.size(1)]


        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature2(fc_feats, att_feats, att_masks)
        state = self.init_hidden(batch_size)
        seq2 = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs2 = fc_feats.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            xt = self.embed(it)

            output, state = self.core2(xt, p_fc_feats, p_att_feats, pp_att_feats, state, p_att_masks, hiddens,
                                       p_hiddens, prev_hidden_masks)
            logprobs = F.log_softmax(self.logit2(output), dim=1)
            # logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq1[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)  # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq2[:, t] = it
            seqLogprobs2[:, t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break
        return seq2, seqLogprobs2


class AdaAtt_lstm(nn.Module):
    def __init__(self, opt, use_maxout=True):
        super(AdaAtt_lstm, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_maxout = use_maxout

        # Build a LSTM
        self.w2h = nn.Linear(self.input_encoding_size, (4+(use_maxout==True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size)

        self.i2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers)])

        # Layers for getting the fake region
        if self.num_layers == 1:
            self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
            self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        else:
            self.r_i2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)


    def forward(self, xt, img_fc, state):

        hs = []
        cs = []
        for L in range(self.num_layers):
            # c,h from previous timesteps
            prev_h = state[0][L]
            prev_c = state[1][L]
            # the input to this layer
            if L == 0:
                x = xt
                i2h = self.w2h(x) + self.v2h(img_fc)
            else:
                x = hs[-1]
                x = F.dropout(x, self.drop_prob_lm, self.training)
                i2h = self.i2h[L-1](x)

            all_input_sums = i2h+self.h2h[L](prev_h)

            sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
            sigmoid_chunk = F.sigmoid(sigmoid_chunk)
            # decode the gates
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
            # decode the write inputs
            if not self.use_maxout:
                in_transform = F.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))
            else:
                in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
                in_transform = torch.max(\
                    in_transform.narrow(1, 0, self.rnn_size),
                    in_transform.narrow(1, self.rnn_size, self.rnn_size))
            # perform the LSTM update
            next_c = forget_gate * prev_c + in_gate * in_transform
            # gated cells form the output
            tanh_nex_c = F.tanh(next_c)
            next_h = out_gate * tanh_nex_c
            if L == self.num_layers-1:
                if L == 0:
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    i2h = self.r_i2h(x)
                n5 = i2h+self.r_h2h(prev_h)
                fake_region = F.sigmoid(n5) * tanh_nex_c

            cs.append(next_c)
            hs.append(next_h)

        # set up the decoder
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)

        state = (torch.cat([_.unsqueeze(0) for _ in hs], 0), 
                torch.cat([_.unsqueeze(0) for _ in cs], 0))
        return top_h, fake_region, state

class AdaAtt_attention(nn.Module):
    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size

        # fake region embed
        self.fr_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.ReLU(), 
            nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        # h out embed
        self.ho_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.Tanh(), 
            nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed, att_masks=None):

        # View into three dimensions
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)

        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)

        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)

        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))

        img_all = torch.cat([fake_region.view(-1,1,self.input_encoding_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1,1,self.input_encoding_size), conv_feat_embed], 1)

        hA = F.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA,self.drop_prob_lm, self.training)
        
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1), dim=1)

        if att_masks is not None:
            att_masks = att_masks.view(-1, att_size)
            PI = PI * torch.cat([att_masks[:,:1], att_masks], 1) # assume one one at the first time step.
            PI = PI / PI.sum(1, keepdim=True)

        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)

        atten_out = visAttdim + h_out_linear

        h = F.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h

class AdaAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaAtt_lstm(opt, use_maxout)
        self.attention = AdaAtt_attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        h_out, p_out, state = self.lstm(xt, fc_feats, state)
        atten_out = self.attention(h_out, p_out, att_feats, p_att_feats, att_masks)
        return atten_out, state

class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


############################################################################
# Notice:
# StackAtt and DenseAtt are models that I randomly designed.
# They are not related to any paper.
############################################################################

from .FCModel import LSTMCore
class StackAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(StackAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt) # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt,fc_feats],1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0,att_res_1],1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([h_1,att_res_2],1), [state[0][2:3], state[1][2:3]])

        return h_2, [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]

class DenseAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(DenseAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt) # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        # fuse h_0 and h_1
        self.fusion1 = nn.Sequential(nn.Linear(opt.rnn_size*2, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))
        # fuse h_0, h_1 and h_2
        self.fusion2 = nn.Sequential(nn.Linear(opt.rnn_size*3, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt,fc_feats],1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0,att_res_1],1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([self.fusion1(torch.cat([h_0, h_1], 1)),att_res_2],1), [state[0][2:3], state[1][2:3]])

        return self.fusion2(torch.cat([h_0, h_1, h_2], 1)), [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

class Att2in2Core(nn.Module):
    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
            self.a2c(att_res)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state


class Att2inCore(Att2in2Core):
    def __init__(self, opt):
        super(Att2inCore, self).__init__(opt)
        del self.a2c
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)

class AdaptiveGate(nn.Module):
    def __init__(self, c1_size, c2_size, h_size, hidden_size):
        super(AdaptiveGate, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(c1_size, self.hidden_size)
        self.W2 = nn.Linear(c2_size, self.hidden_size)
        self.Wh = nn.Linear(h_size, self.hidden_size)
        #self.proj = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, c1, c2, h):
        gamma = F.sigmoid(self.W1(c1) + self.W2(c2) + self.Wh(h))
        c = gamma * c1 + (1 - gamma) * c2
        return c#self.proj(torch.cat([c, h], 1))

class Att2inPairCore(nn.Module):
    def __init__(self, opt):
        super(Att2inPairCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        # self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        # Build a LSTM
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

        self.prev_attention = Attention(opt)
        self.adaptive_gate = AdaptiveGate(self.att_feat_size, self.att_feat_size, self.rnn_size, self.att_feat_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks, prev_states, p_prev_states, prev_states_masks):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)
        att_prev = self.prev_attention(state[0][-1], prev_states, p_prev_states, prev_states_masks)
        att_res = self.adaptive_gate(att_res, att_prev, state[0][-1])
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
                       self.a2c(att_res)
        in_transform = torch.max( \
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class Att2in2PairCore(Att2inPairCore):
    def __init__(self, opt):
        super(Att2in2PairCore, self).__init__(opt)
        del self.a2c, self.adaptive_gate
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.adaptive_gate = AdaptiveGate(self.rnn_size, self.rnn_size, self.rnn_size, self.rnn_size)

class Att2inPair2Core(nn.Module):
    def __init__(self, opt):
        super(Att2inPair2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        # self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        # self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        # Build a LSTM
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

        self.prev_attention = Attention(opt)
        self.attention_map = nn.Linear(self.att_feat_size*2, self.att_feat_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks, prev_states, p_prev_states, prev_states_masks):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)
        att_prev = self.prev_attention(state[0][-1], prev_states, p_prev_states, prev_states_masks)
        att_res = self.attention_map(torch.cat([att_res, att_prev], 1).contiguous())
        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
                       self.a2c(att_res)
        in_transform = torch.max( \
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

"""
Note this is my attempt to replicate att2all model in self-critical paper.
However, this is not a correct replication actually. Will fix it.
"""
class Att2all2Core(nn.Module):
    def __init__(self, opt):
        super(Att2all2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1]) + self.a2h(att_res)
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * F.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class Att2inPairModel(AttPairModel):
    def __init__(self, opt):
        super(Att2inPairModel, self).__init__(opt)
        del self.embed, self.fc_embed, self.att_embed
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.fc_embed = self.att_embed = lambda x: x
        del self.ctx2att
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.ctx2att2 = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.core = Att2inCore(opt)
        self.core2 = Att2inPairCore(opt)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)
        self.logit2.bias.data.fill_(0)
        self.logit2.weight.data.uniform_(-initrange, initrange)


class Att2in2PairModel(AttPairModel):
    def __init__(self, opt):
        super(Att2in2PairModel, self).__init__(opt)
        self.core = Att2in2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x
        self.core2 = Att2in2PairCore(opt)
        del self.prev_sent_wrap, self.prev_sent_emb
        self.prev_sent_emb = nn.Linear(self.rnn_size, self.rnn_size)
        self.prev_sent_wrap = nn.Linear(self.rnn_size, self.att_hid_size)


class Att2inPair2Model(AttPairModel):
    def __init__(self, opt):
        super(Att2inPair2Model, self).__init__(opt)
        del self.embed, self.fc_embed, self.att_embed
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.fc_embed = self.att_embed = lambda x: x
        del self.ctx2att
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.ctx2att2 = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.core = Att2inCore(opt)
        self.core2 = Att2inPair2Core(opt)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)
        self.logit2.bias.data.fill_(0)
        self.logit2.weight.data.uniform_(-initrange, initrange)
