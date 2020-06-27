from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models

import eval_utils_pair
import misc.utils as utils
import sys
from misc.rewards import init_scorer, get_self_critical_reward
import numpy as np


try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def train(opt):
    import random
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    if opt.use_box: opt.att_feat_size = opt.att_feat_size + 5

    from dataloader_pair import DataLoader

    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    if opt.log_to_file:
        if os.path.exists(os.path.join(opt.checkpoint_path, 'log')):
            suffix = time.strftime("%Y-%m-%d %X", time.localtime())
            print('Warning !!! %s already exists ! use suffix ! ' %os.path.join(opt.checkpoint_path, 'log'))
            sys.stdout = open(os.path.join(opt.checkpoint_path, 'log' + suffix), "w")
        else:
            print('logging to file %s'%os.path.join(opt.checkpoint_path, 'log'))
            sys.stdout = open(os.path.join(opt.checkpoint_path, 'log'), "w")

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        if os.path.isfile(opt.start_from):
            with open(os.path.join(opt.infos)) as f:
                infos = cPickle.load(f)
                saved_model_opt = infos['opt']
                need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
                for checkme in need_be_same:
                    assert vars(saved_model_opt)[checkme] == vars(opt)[
                        checkme], "Command line argument and saved model disagree on '%s' " % checkme
        else:
            if opt.load_best != 0:
                print('loading best info')
                with open(os.path.join(opt.start_from, 'infos_' + opt.id + '-best.pkl')) as f:
                    infos = cPickle.load(f)
                    saved_model_opt = infos['opt']
                    need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
                    for checkme in need_be_same:
                        assert vars(saved_model_opt)[checkme] == vars(opt)[
                            checkme], "Command line argument and saved model disagree on '%s' " % checkme
            else:
                with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')) as f:
                    infos = cPickle.load(f)
                    saved_model_opt = infos['opt']
                    need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
                    for checkme in need_be_same:
                        assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
                try:
                    histories = cPickle.load(f)
                except:
                    print('load history error!')
                    histories = {}

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    start_epoch = epoch

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt).cuda()
    dp_model = torch.nn.DataParallel(model)

    update_lr_flag = True
    # Assure in training mode
    dp_model.train()

    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    optimizer = utils.build_optimizer(model.parameters(), opt)
    #Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    if opt.caption_model == 'att2in2p':
        optimized = ['logit2', 'ctx2att2', 'core2', 'prev_sent_emb', 'prev_sent_wrap']
        optimized_param = []
        optimized_param1 = []

        for name, param in model.named_parameters():
            second = False
            for n in optimized:
                if n in name:
                    print('second', name)
                    optimized_param.append(param)
                    second = True
            if 'embed' in name:
                print('all', name)
                optimized_param1.append(param)
                optimized_param.append(param)
            elif not second:
                print('first', name)
                optimized_param1.append(param)

    while True:
        if opt.val_only:
            eval_kwargs = {'split': 'val',
                           'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            print('start evaluating')
            val_loss, predictions, lang_stats = eval_utils_pair.eval_split(dp_model, crit, loader, eval_kwargs)
            exit(0)
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False
                
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['pair_fc_feats'], data['pair_att_feats'], data['pair_labels'], data['pair_masks'], data['pair_att_masks']]

        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        masks = masks.float()

        optimizer.zero_grad()

        if not sc_flag:
            if opt.onlysecond:
                # only using the second sentence from a visual paraphrase pair. opt.caption_model should be a one-stage decoding model
                loss = crit(dp_model(fc_feats, att_feats, labels[:, 1, :], att_masks), labels[:, 1, 1:], masks[:, 1, 1:])
                loss1 = loss2 = loss/2
            elif opt.first:
                # using the first sentence
                tmp = [data['first_fc_feats'], data['first_att_feats'], data['first_labels'], data['first_masks'],
                       data['first_att_masks']]
                tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks = tmp
                masks = masks.float()
                loss = crit(dp_model(fc_feats, att_feats, labels[:, :], att_masks), labels[:, 1:],
                            masks[:, 1:])
                loss1 = loss2 = loss / 2
            elif opt.onlyfirst:
                # only using the second sentence from a visual paraphrase pair
                loss = crit(dp_model(fc_feats, att_feats, labels[:, 0, :], att_masks), labels[:, 0, 1:],
                            masks[:, 0, 1:])
                loss1 = loss2 = loss / 2
            else:
                # proposed DCVP model, opt.caption_model should be att2inp
                output1, output2 = dp_model(fc_feats, att_feats, labels, att_masks, masks[:, 0, 1:])
                loss1 = crit(output1, labels[:, 0, 1:], masks[:, 0, 1:])
                loss2 = crit(output2, labels[:, 1, 1:], masks[:, 1, 1:])
                loss = loss1 + loss2

        else:
            raise NotImplementedError
            # Our DCVP model does not support self-critical sequence training
            # We found that RL(SCST) with CIDEr reward will improve conventional metrics (BLEU, CIDEr, etc.)
            # but harm diversity and descriptiveness
            # Please refer to the paper for the details

        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()

        train_loss = loss.item()
        torch.cuda.synchronize()
        end = time.time()
        if not sc_flag:
            print("iter {} (epoch {}), train_loss = {:.3f}, loss1 = {:.3f}, loss2 = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, loss.item(), loss1.item(), loss2.item(), end - start))
        else:
            print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, np.mean(reward[:,0]), end - start))

        sys.stdout.flush()
        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils_pair.eval_split(dp_model, crit, loader, eval_kwargs)

            # Write validation result into summary
            add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k,v in lang_stats.items():
                    add_summary_value(tb_summary_writer, k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model' + str(iteration) + '.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '_' + str(iteration) + '.pkl'),
                          'wb') as f:
                    cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
train(opt)
