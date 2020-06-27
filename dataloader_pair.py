from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random

import torch
import torch.utils.data as data

import multiprocessing

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        
        # feature related options
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        
        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
        self.pair_h5 = h5py.File(opt.pair_h5, 'r', driver='core')

        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        self.input_box_dir = self.opt.input_box_dir

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}
        
        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = [] # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = [] # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')

        wrapped = False

        infos = []
        gts = []

        ixes = []

        for i in range(batch_size):
            # fetch image
            tmp_fc, tmp_att,\
                ix, tmp_wrapped = self._prefetch_process[split].get()
            ixes.append(ix)
            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)
            
            label_batch[i * seq_per_img : (i + 1) * seq_per_img, 1 : self.seq_length + 1] = self.get_captions(ix, seq_per_img)

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
        
            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)


        # Get pair caption data
        data = dict()
        pair_gts = None

        if split == 'train':
            all_pairs = []
            img_ixes = []
            rank_masks = []
            rank_pairs = []

            first_seqs = []
            img_first_ix = []

            weights = {}
            first_gts = []
            pair_gts = []
            for i, ix in enumerate(ixes):
                pair_data = np.array(self.pair_h5[str(ix)])  # num_pairs * 2
                ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
                ix2 = self.label_end_ix[ix] - 1
                seqs = self.h5_label_file['labels'][ix1: ix2 + 1]

                for i_pair in range(pair_data.shape[0]):
                    first_sent = seqs[int(pair_data[i_pair, 0])]  # (16,)
                    seq_len = first_sent.shape[0]
                    second_sent = seqs[int(pair_data[i_pair, 1])]
                    all_pairs.append(
                        np.concatenate([first_sent[:, np.newaxis], second_sent[:, np.newaxis]], 1))
                    img_ixes.append(i)

                    if pair_data.shape[1] > 2:
                        rank_sents = np.zeros([pair_data.shape[1] - 2, seq_len])
                        rank_mask = np.zeros(pair_data.shape[1] - 2)
                        for k in range(2, pair_data.shape[1]):
                            if int(pair_data[i_pair, k]) >= 0:
                                rank_sents[k-2] = seqs[int(pair_data[i_pair, k])]
                                rank_mask[k-2] = 1
                        rank_pairs.append(rank_sents)
                        rank_masks.append(rank_mask)

                gt_dict = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}
                first_gt = []
                for i_pair in range(pair_data.shape[0]):
                    gt_dict[int(pair_data[i_pair, 0])] += [int(pair_data[i_pair, 1])]
                    if int(pair_data[i_pair, 0]) not in first_gt:
                        first_gt.append(int(pair_data[i_pair, 0]))
                if first_gt: # first_gt may be []
                    first_gt = np.concatenate([seqs[_][np.newaxis, :] for _ in first_gt], 0)

                gt_sent_dict = {}
                for k, v in gt_dict.items():
                    if v:
                        gt_sent_dict[k] = np.concatenate([seqs[_][np.newaxis, :] for _ in v], 0)

                for i_pair in range(pair_data.shape[0]):
                    pair_gts.append(gt_sent_dict[int(pair_data[i_pair, 0])])
                    #pair_gts.append(seqs)
                    first_gts.append(first_gt)
                # for it in pair_data[:, 1]:
                #     weight[it] += 1
                # weights[ix] = weight

                first_ixes = []
                for i_pair in range(pair_data.shape[0]):
                    if int(pair_data[i_pair, 0]) not in first_ixes:
                        first_ixes.append(int(pair_data[i_pair, 0]))
                for first_ix in first_ixes:
                    first_seqs.append(seqs[int(first_ix)])
                    img_first_ix.append(i)

            all_pairs = np.stack(all_pairs, axis=2).transpose(2, 1, 0)
            first_seqs = np.stack(first_seqs, axis=0)
            first_labels = np.zeros([first_seqs.shape[0], first_seqs.shape[1] + 2])
            first_labels[:, 1:-1] = first_seqs
            new_pairs = np.zeros([all_pairs.shape[0], 2, all_pairs.shape[2]+2])
            new_pairs[:, :, 1:-1] = all_pairs

            if rank_pairs:
                rank_pairs = np.stack(rank_pairs, axis=0)
                rank_masks = np.stack(rank_masks, axis=0)

                new_rank_pairs = np.zeros([rank_pairs.shape[0], rank_pairs.shape[1], rank_pairs.shape[2]+2])
                new_rank_pairs[:, :, 1:-1] = rank_pairs

                data['rank_sent_masks'] = (new_rank_pairs > 0).astype(np.float)
                for b in range(data['rank_sent_masks'].shape[0]):
                    for d in range(data['rank_sent_masks'].shape[1]):
                        # for <eos>
                        for c in range(data['rank_sent_masks'].shape[2] - 1):
                            if data['rank_sent_masks'][b, d, c] == 1.0 and data['rank_sent_masks'][b, d, c + 1] == 0.0:
                                data['rank_sent_masks'][b, d, c + 1] = 1.0
                                break
                data['rank_sent_masks'][:, :, 1] = 1
                data['rank_masks'] = rank_masks
                data['rank_pairs'] = new_rank_pairs.astype(np.long)
                assert rank_pairs.shape[0] == rank_masks.shape[0] == new_pairs.shape[0]


            max_att_len = max([_.shape[0] for _ in att_batch])
            data['pair_att_feats'] = np.zeros([len(img_ixes), max_att_len, att_batch[0].shape[1]], dtype='float32')
            for i in range(len(img_ixes)):
                data['pair_att_feats'][i, :att_batch[img_ixes[i]].shape[0]] = att_batch[img_ixes[i]]
            data['pair_fc_feats'] = np.zeros([len(img_ixes), fc_batch[0].shape[0]],
                                              dtype='float32')
            for i in range(len(img_ixes)):
                data['pair_fc_feats'][i] = fc_batch[img_ixes[i]]
            data['pair_att_masks'] = np.zeros(data['pair_att_feats'].shape[:2], dtype='float32')
            for i in range(len(data['pair_att_feats'])):
                data['pair_att_masks'][i, :att_batch[img_ixes[i]].shape[0]] = 1
            # set att_masks to None if attention features have same length
            if data['pair_att_masks'].sum() == data['pair_att_masks'].size:
                data['pair_att_masks'] = None
            data['pair_labels'] = new_pairs.astype(np.long)
            # generate mask

            data['pair_masks'] = (data['pair_labels'] > 0).astype(np.float)
            for b in range(data['pair_masks'].shape[0]):
                for d in range(0, 2):
                    # for <eos>
                    for c in range(data['pair_masks'].shape[2] - 1):
                        if data['pair_masks'][b,d,c] == 1.0 and data['pair_masks'][b,d, c+1] == 0.0:
                            data['pair_masks'][b, d, c+1] = 1.0
                            break

            max_att_len = max([_.shape[0] for _ in att_batch])
            data['first_att_feats'] = np.zeros([len(img_first_ix), max_att_len, att_batch[0].shape[1]], dtype='float32')
            for i in range(len(img_first_ix)):
                data['first_att_feats'][i, :att_batch[img_first_ix[i]].shape[0]] = att_batch[img_first_ix[i]]
            data['first_fc_feats'] = np.zeros([len(img_first_ix), fc_batch[0].shape[0]],
                                             dtype='float32')
            for i in range(len(img_first_ix)):
                data['first_fc_feats'][i] = fc_batch[img_first_ix[i]]
            data['first_att_masks'] = np.zeros(data['first_att_feats'].shape[:2], dtype='float32')
            for i in range(len(data['first_att_feats'])):
                data['first_att_masks'][i, :att_batch[img_first_ix[i]].shape[0]] = 1
            # set att_masks to None if attention features have same length
            if data['first_att_masks'].sum() == data['first_att_masks'].size:
                data['first_att_masks'] = None
            data['first_labels'] = first_labels.astype(np.long)
            # generate mask

            data['first_masks'] = (data['first_labels'] > 0).astype(np.float)
            for b in range(data['first_masks'].shape[0]):
                    # for <eos>
                    for c in range(data['first_masks'].shape[1] - 1):
                        if data['first_masks'][b, c] == 1.0 and data['first_masks'][b, c + 1] == 0.0:
                            data['first_masks'][b, c + 1] = 1.0
                            break

            data['caption_weights'] = weights
        data['fc_feats'] = np.stack(reduce(lambda x, y: x + y, [[_] * seq_per_img for _ in fc_batch]))
        # merge att_feats
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch) * seq_per_img, max_att_len, att_batch[0].shape[1]],
                                     dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i * seq_per_img:(i + 1) * seq_per_img, :att_batch[i].shape[0]] = 1
        # set att_masks to None if attention features have same length
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch
        if pair_gts is not None:
            data['second_gts'] = pair_gts
            data['first_gts'] = first_gts
        else:
            data['gts'] = gts # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index #self.split_ix[index]
        if self.use_att:
            att_feat = np.load(os.path.join(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
            # Reshape to K x C
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = np.load(os.path.join(self.input_box_dir, str(self.info['images'][ix]['id']) + '.npy'))
                # devided by image width and height
                x1,y1,x2,y2 = np.hsplit(box_feat, 4)
                h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
                if self.norm_box_feat:
                    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                att_feat = np.hstack([att_feat, box_feat])
                # sort the features by the size of boxes
                att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
            fc_feat = np.zeros(1)
        else:
            att_feat = np.zeros((1,1,1))
            fc_feat = np.load(os.path.join(self.input_fc_dir, str(self.info['images'][ix]['id']) + '.npy'))
        return (fc_feat,
                att_feat,
                ix)

    def __len__(self):
        return len(self.info['images'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=1, # 4 is usually enough
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[2] == ix, "ix not equal"

        return tmp + [wrapped]