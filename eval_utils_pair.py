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
import models
import cPickle
from dataloader_pair import DataLoader
import torchvision.transforms as transforms
from PIL import Image

def get_transform(data_name, split_name, opt):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    t_list = []
    if split_name == 'train':
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    elif split_name == 'val':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    elif split_name == 'test':
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    t_end = [transforms.ToTensor(), normalizer]
    transform = transforms.Compose(t_list + t_end)
    return transform

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
        preds_filt = preds  # [p for p in preds if p['image_id'] in valids]
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

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    predictions_1 = []
    seqs = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size
        sys.stdout.flush()
        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample

        # forward the model to also get generated samples for each image
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
               data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data[
                                                                                           'att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, att_masks = tmp
        with torch.no_grad():
            if eval_kwargs.get('onlysecond', 0) or eval_kwargs.get('caption_model', 0) == 'att2in':
                seq1 = seq2 = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data
            elif eval_kwargs.get('onlyfirst', 0) or eval_kwargs.get('caption_model', 0) == 'att2in':
                seq1 = seq2 = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data
            elif eval_kwargs.get('first', 0) or eval_kwargs.get('caption_model', 0) == 'att2in':
                seq1 = seq2 = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data
            else:
                seq1, seq2 = model(fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                pass#print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                #print('--' * 10)
        sents1 = utils.decode_sequence(loader.get_vocab(), seq1)
        sents2 = utils.decode_sequence(loader.get_vocab(), seq2)
        for k, sent in enumerate(sents2):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'first_caption': sents1[k]}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print(data['infos'][k]['file_path'])
                print('image %s caption1: %s' %(entry['image_id'], entry['first_caption']))
                print('image %s caption2: %s' % (entry['image_id'], entry['caption']))

            entry_1 = {'image_id': data['infos'][k]['id'], 'caption': sents1[k]}
            predictions_1.append(entry_1)

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        # if n > ix1:
        #     seq2 = seq2[:(ix1 - n) * loader.seq_per_img]
        if eval_kwargs.get('vsepp', 0):
            if len(seq2.size()) == 2:
                seqs.append(padding(seq2, 30))
        for i in range(n - ix1):
            predictions.pop()
            predictions_1.pop()
        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    if eval_kwargs.get('vsepp', 0):
        seqs = torch.cat(seqs, 0).contiguous()
        # print(seqs)
        seqs = change_seq(seqs, loader.ix_to_word)
        seqs = seqs[:num_images]

        from eval_vsepp import evalrank_vsepp
        imgids = [_['image_id'] for _ in predictions]
        seqs = seqs[:num_images]

        transform = get_transform('COCO', 'val', None)
        imgs = []
        for i, imgid in enumerate(imgids):
            # raw jpg image paths
            img_path = '../imgcap/data/raw_images/val2014/COCO_val2014_'+str(imgid).zfill(12)+'.jpg'
            if i % 100 == 0:
                print('load %d images' %i)
            image = Image.open(img_path).convert('RGB')
            image = transform(image)
            imgs.append(image.unsqueeze(0))
        imgs = torch.cat(imgs, 0).contiguous()
        lengths = torch.sum((seqs > 0).long(), 1) + 1
        lengths = lengths.cpu()
        with torch.no_grad():
            evalrank_vsepp(imgs, loader.ix_to_word, seqs, lengths)

    if lang_eval == 1:
        if eval_kwargs.get('val_only', 0):
            lang_stats = language_eval(eval_kwargs.get('data', 'coco'), predictions, eval_kwargs['id'], split + 'val_only')
        else:
            lang_stats = language_eval(eval_kwargs.get('data', 'coco'), predictions, eval_kwargs['id'], split)

        if eval_kwargs.get('onlyfirst', 0) == 0 and eval_kwargs.get('first', 0) == 0:
            lang_stats_1 = language_eval(eval_kwargs.get('data', 'coco'), predictions_1, eval_kwargs['id'],
                                         split + '_caption1')
            for k, v in lang_stats_1.items():
                lang_stats[k + '_caption1'] = v

    if eval_kwargs.get('rank', 0):
        ranks = evalrank(rank_model, loader, seqs, eval_kwargs)



    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


def encode_data(model, loader, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    model.eval()

    loader_seq_per_img = loader.seq_per_img
    loader.seq_per_img = 5
    loader.reset_iterator(split)

    n = 0
    img_embs = []
    cap_embs = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        tmp = utils.var_wrapper(tmp, volatile=True)
        fc_feats, att_feats, labels, masks = tmp
        labels[:, 0] = len(loader.ix_to_word) + 1
        img_emb = model.vse.img_enc(fc_feats)
        cap_emb = model.vse.txt_enc(labels, masks)

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)

        if n > ix1:
            img_emb = img_emb[:(ix1 - n) * loader.seq_per_img]
            cap_emb = cap_emb[:(ix1 - n) * loader.seq_per_img]

        # preserve the embeddings by copying from gpu and converting to np
        img_embs.append(img_emb.data.cpu().numpy().copy())
        cap_embs.append(cap_emb.data.cpu().numpy().copy())

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

        print("%d/%d" % (n, ix1))

    img_embs = np.vstack(img_embs)
    cap_embs = np.vstack(cap_embs)
    assert img_embs.shape[0] == ix1 * loader.seq_per_img

    loader.seq_per_img = loader_seq_per_img

    return img_embs, cap_embs


def encode_data_generated(model, loader, captions, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')

    # Make sure in the evaluation mode
    model.eval()

    loader_seq_per_img = loader.seq_per_img
    loader.seq_per_img = 5
    loader.reset_iterator(split)
    print('num_images', num_images)
    print(captions.size())
    print(len(loader))
    n = 0
    img_embs = []
    cap_embs = []
    while True:
        data = loader.get_batch(split)
        labels = captions[n: (n + loader.batch_size)]
        masks = (labels > 0).float()
        for i in range(labels.size(0)):
            for j in range(labels.size(1) - 1):
                if labels[i, j].item() > 0.5 and labels[i, j+1].item() < 0.5:
                    masks[i, j+1] = 1.0
                    break

        n = n + loader.batch_size

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        tmp = utils.var_wrapper(tmp, volatile=True)
        fc_feats, att_feats, _, __ = tmp
        fc_feats = fc_feats.cuda()
        labels = labels.cuda()
        masks = masks.cuda()
        img_emb = model.vse.img_enc(fc_feats)
        cap_emb = model.vse.txt_enc(labels, masks)

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)

        print(img_emb.size())
        img_embs.append(img_emb.data.cpu().numpy().copy())
        cap_embs.append(cap_emb.data.cpu().numpy().copy())
        if n > ix1:
            img_emb = img_emb[:(ix1 - n) * loader.seq_per_img]
            cap_emb = cap_emb[:(ix1 - n) * loader.seq_per_img]
        # preserve the embeddings by copying from gpu and converting to np

        print(cap_emb.size())
        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

        print("%d/%d" % (n, ix1))
    print('start stack')
    img_embs = np.vstack(img_embs)[:num_images * 5]
    cap_embs = np.vstack(cap_embs)[:num_images]
    print(img_embs.shape)
    print(cap_embs.shape)
    print('stack')

    #assert img_embs.shape[0] == ix1 * loader.seq_per_img

    loader.seq_per_img = loader_seq_per_img

    return img_embs, cap_embs


def evalrank(model, loader, captions=None, eval_kwargs={}):
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    dataset = eval_kwargs.get('dataset', 'coco')
    fold5 = eval_kwargs.get('fold5', 0)
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    print('Computing results...')
    if captions is not None:
        print('Encoding generated data')
        img_embs, cap_embs = encode_data_generated(model, loader, captions, eval_kwargs)
    else:
        img_embs, cap_embs = encode_data(model, loader, eval_kwargs)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))
    print('data is encoded')
    if not fold5:
        # no cross-validation, full evaluation
        if captions is not None:
            ri, rti = t2i_generated(img_embs, cap_embs,
                          measure='cosine', return_ranks=True)
            rsum = ri[0] + ri[1] + ri[2]
        else:
            r, rt = i2t(img_embs, cap_embs, measure='cosine', return_ranks=True)
            ri, rti = t2i(img_embs, cap_embs,
                      measure='cosine', return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("Average i2t Recall: %.1f" % ar)
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        ari = (ri[0] + ri[1] + ri[2]) / 3
        print("rsum: %.1f" % rsum)

        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            r, rt0 = i2t(img_embs[i * 5000:(i + 1) * 5000],
                         cap_embs[i * 5000:(i + 1) *
                                           5000], measure='cosine',
                         return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs[i * 5000:(i + 1) * 5000],
                           cap_embs[i * 5000:(i + 1) *
                                             5000], measure='cosine',
                           return_ranks=True)
            if i == 0:
                rt, rti = rt0, rti0
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    return {'t2i_r1': ri[0], 't2i_r5': ri[1], 't2i_r10': ri[2], 't2i_medr': ri[3],
            't2i_meanr': ri[4]}
        # #{'rsum': rsum, 'i2t_ar': ar, 't2i_ar': ari,
        #     'i2t_r1': r[0], 'i2t_r5': r[1], 'i2t_r10': r[2], 'i2t_medr': r[3], 'i2t_meanr': r[4],
        #     't2i_r1': ri[0], 't2i_r5': ri[1], 't2i_r10': ri[2], 't2i_medr': ri[3],
        #     't2i_meanr': ri[4]}  # {'rt': rt, 'rti': rti}


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        if measure == 'order':
            bs = 100
            if index % bs == 0:
                mx = min(images.shape[0], 5 * (index + bs))
                im2 = images[5 * index:mx:5]
                d2 = order_sim(torch.Tensor(im2).cuda(),
                               torch.Tensor(captions).cuda())
                d2 = d2.cpu().numpy()
            d = d2[index % bs]
        else:
            d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = images.shape[0] // 5
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i_generated(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    print(images.shape, captions.shape)
    if npts is None:
        npts = images.shape[0]  // 5
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query captions
        queries = captions[index: index + 1].reshape(1, captions.shape[1]) # only one query
        # Compute scores
        if measure == 'order':
            bs = 100
            if 5 * index % bs == 0:
                mx = min(captions.shape[0], 5 * index + bs)
                q2 = captions[5 * index:mx]
                d2 = order_sim(torch.Tensor(ims).cuda(),
                               torch.Tensor(q2).cuda())
                d2 = d2.cpu().numpy()

            d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        else:
            d = np.dot(queries, ims.T)

        inds = np.argsort(d[0])[::-1]
        ranks[index] = np.where(inds == index)[0][0]
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
