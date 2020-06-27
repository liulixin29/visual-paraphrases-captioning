from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
import cPickle
import models

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    if dataset == 'coco':
        annFile = '../imgcap/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    if dataset == 'coco':
        preds_filt = preds#[p for p in preds if p['image_id'] in valids]
    else:
        preds_filt = preds
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    # imgToEval = cocoEval.imgToEval
    # for p in preds_filt:
    #     image_id, caption = p['image_id'], p['caption']
    #     imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out}, outfile) #, 'imgToEval': imgToEval

    return out


def padding(input, size):
    pad_len = size - input.size(1)
    input = torch.cat([input, input.new(input.size(0), pad_len).zero_()], 1).contiguous()
    return input

def change_seq(input, ixtoword):
    bos = input.new(input.size(0), 1).zero_()
    bos[:] = len(ixtoword) + 1
    output = torch.cat([bos, input], 1).contiguous()
    return output


def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)



    if eval_kwargs.get('rank', 0):
        infos_path = 'log_fc_con/infos_vse_fc_con-best.pkl'  # 'log_fc_con_discsplit/infos_vse_fc_con_discsplit-best.pkl'
        model_path = 'log_fc_con/model_vse-best.pth'  # 'log_fc_con_discsplit/model_vse-best.pth'
        with open(infos_path) as f:
            infos = cPickle.load(f)

        rank_model = models.JointModel(infos['opt'])
        utils.load_state_dict(rank_model, torch.load(model_path))
        rank_model.cuda()
        rank_model.eval()
        print('success loaded retrieval model !')

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    seqs = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        sys.stdout.flush()
        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data
        
        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                pass#print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                #print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if eval_kwargs.get('rank', 0):
            seqs.append(padding(seq, 30))
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()
        if n > ix1:
            seq = seq[:(ix1 - n) * loader.seq_per_img]

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    if eval_kwargs.get('rank', 0):
        seqs = torch.cat(seqs, 0).contiguous()
        seqs = change_seq(seqs, loader.ix_to_word)

    if eval_kwargs.get('vsepp', 0):
        from eval_vsepp import evalrank_vsepp
        from eval_utils_pair import get_transform
        import torchvision.transforms as transforms
        from PIL import Image

        imgids = [_['image_id'] for _ in predictions]
        seqs = seqs[:num_images]

        transform = get_transform('COCO', 'val', None)
        imgs = []
        for i, imgid in enumerate(imgids):
            img_path = '../imgcap/data/raw_images/val2014/COCO_val2014_'+str(imgid).zfill(12)+'.jpg'
            if i % 100 == 0:
                print('load %d images' %i)
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            imgs.append(image.unsqueeze(0))
        imgs = torch.cat(imgs, 0).contiguous()
        lengths = torch.sum((seqs > 0), 1) + 1
        lengths = lengths.cpu()
        with torch.no_grad():
            evalrank_vsepp(imgs, loader.ix_to_word, seqs, lengths)

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(eval_kwargs.get('data', 'coco'), predictions, eval_kwargs['id'], split)

    if eval_kwargs.get('rank', 0):
        ranks = evalrank(rank_model, loader, seqs, eval_kwargs)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


