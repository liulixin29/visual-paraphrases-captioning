from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .OldModel import ShowAttendTellModel, AllImgModel
# from .Att2inModel import Att2inModel
from .AttModel import *
from .AttPairModel import Att2inPairModel, Att2inPair2Model, Att2in2PairModel
from .JointModel import JointModel
from .beam import Beam


def setup(opt):
    if opt.caption_model == 'fc':
        model = FCModel(opt)
    elif opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    # Att2in model in self-critical
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    elif opt.caption_model == 'att2inp':
        model = Att2inPairModel(opt)
    elif opt.caption_model == 'att2in2p':
        model = Att2in2PairModel(opt)
    elif opt.caption_model == 'att2inp2':
        model = Att2inPair2Model(opt)
    elif opt.caption_model == 'sep':
        model = Att2inPairSepModel(opt)
    elif opt.caption_model == 'pairn':
        model = Att2inPairNModel(opt)
    # Att2in model with two-layer MLP img embedding and word embedding
    elif opt.caption_model == 'att2in2':
        model = Att2in2Model(opt)
    elif opt.caption_model == 'att2all2':
        model = Att2all2Model(opt)
    # Adaptive Attention model from Knowing when to look
    elif opt.caption_model == 'adaatt':
        model = AdaAttModel(opt)
    # Adaptive Attention with maxout lstm
    elif opt.caption_model == 'adaattmo':
        model = AdaAttMOModel(opt)
    # Top-down attention model
    elif opt.caption_model == 'topdown':
        model = TopDownModel(opt)
    # StackAtt
    elif opt.caption_model == 'stackatt':
        model = StackAttModel(opt)
    # DenseAtt
    elif opt.caption_model == 'denseatt':
        model = DenseAttModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.exists(opt.start_from)," %s must be a a path" % opt.start_from
        if os.path.isfile(opt.start_from):
            model.load_state_dict(torch.load(opt.start_from))
        else:
            if vars(opt).get('load_best', 0) != 0:

                assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+"-best.pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
                model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-best.pth')))
                print('load best model!')
            else:
                assert os.path.isfile(os.path.join(opt.start_from,
                                                   "infos_" + opt.id + ".pkl")), "infos.pkl file does not exist in path %s" % opt.start_from
                model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model

def load(model, opt):
    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        utils.load_state_dict(model, torch.load(os.path.join(opt.start_from, 'model.pth')))

