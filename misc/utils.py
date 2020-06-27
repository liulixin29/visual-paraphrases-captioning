from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable


def if_use_att(caption_model):
    # Decide if load attention feature according to caption model
    if caption_model in ['show_tell', 'all_img', 'fc']:
        return False
    return True

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class ScoreLanguageModelCriterion(nn.Module):
    def __init__(self):
        super(ScoreLanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, scores):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output, 1) / torch.sum(mask, 1)
        assert output.size(0) == len(scores)
        batch_size = len(scores)
        for i in range(batch_size):
            output[i] = output[i] * scores[i]
        output = torch.sum(output) / batch_size

        return output


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))

def var_wrapper(x, cuda=True, volatile=False):
    if type(x) is dict:
        return {k: var_wrapper(v, cuda, volatile) for k,v in x.items()}
    if type(x) is list or type(x) is tuple:
        return [var_wrapper(v, cuda, volatile) for v in x]
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if cuda:
        x = x.cuda()
    else:
        x = x.cpu()
    if torch.is_tensor(x):
        x = Variable(x, volatile=volatile)
    if isinstance(x, Variable) and volatile!=x.volatile:
        x = Variable(x.data, volatile=volatile)
    return x


def load_state_dict(model, state_dict):
    model_state_dict = model.state_dict()
    keys = set(model_state_dict.keys() + state_dict.keys())
    for k in keys:
        if k not in state_dict:
            print('key %s in model.state_dict() not in loaded state_dict' %(k))
        elif k not in model_state_dict:
            print('key %s in loaded state_dict not in model.state_dict()' %(k))
        else:
            if state_dict[k].size() != model_state_dict[k].size():
                print('key %s size not match in model.state_dict() and loaded state_dict. Try to flatten and copy the values in common parts' %(k))
            model_state_dict[k].view(-1)[:min(model_state_dict[k].numel(), state_dict[k].numel())]\
                .copy_(state_dict[k].view(-1)[:min(model_state_dict[k].numel(), state_dict[k].numel())])

    model.load_state_dict(model_state_dict)