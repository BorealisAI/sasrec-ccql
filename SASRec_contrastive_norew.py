# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2019-present, Wang-Cheng Kang, Julian McAuley.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the SASREC (https://arxiv.org/abs/1808.09781.pdf) implementation
# from https://github.com/kang205/SASRec by Michael Kelly and Julian McAuley
#################################################################################### 

import numpy as np
import torch
import pandas as pd
import os
import argparse
import time
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="SASRecContrastive.")

    parser.add_argument('--epoch',
                        type=int,
                        default=5000,
                        help='Number of max epochs.')
    parser.add_argument('--dataset',
                        nargs='?',
                        default='movielens',
                        help='dataset')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size.')
    parser.add_argument('--maxlen', default=10, type=int)
    parser.add_argument('--hidden_factor',
                        type=int,
                        default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate.')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_interval', default=2000, type=int)
    parser.add_argument('--contrastive_loss',
                        type=str,
                        default='InfoNCE')
    parser.add_argument('--aug', type=str, default='permutation')
    parser.add_argument('--exp_id', type=str, default='SASRec_Contrastive_NoRew')
    parser.add_argument('--console', default=200, type=int)
    parser.add_argument('--seq', type=int, default=30)
    return parser.parse_args()


def calculate_hit_norew(sorted_list, topk, true_items):

    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]

    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]

        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)
    return hit_purchase, ndcg_purchase


def evaluate(model, replay_buffer, total_step):
    batch = 100
    total_clicks = 0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    eval_start = time.time()

    num_rows = replay_buffer.shape[0]
    num_batches = int(num_rows / args.batch_size)

    for j in range(num_batches):
        batch = replay_buffer.sample(n=args.batch_size).to_dict()
        states = list(batch['state'].values())
        len_states = list(batch['len_state'].values())
        actions = list(batch['action'].values())

        states = np.asarray(states)
        len_states = np.asarray(len_states)
        logits = model(states, len_states)
        prediction = model.cls_layer(logits)
        prediction = prediction.detach().cpu().numpy()
        sorted_list = np.argsort(prediction)
        hit_purchase, ndcg_purchase = calculate_hit_norew(
            sorted_list, topk, actions)
    print('#############################################################')
    print('total clicks: %d, total purchase:%d' %
          (total_clicks, total_purchase))
    eval_total_time = time.time() - eval_start
    for i in range(len(topk)):
        try:
            hr_purchase = hit_purchase[i] / args.batch_size
        except:
            hr_purchase = 0
        try:
            ng_purchase = ndcg_purchase[i] / args.batch_size
        except:
            ng_purchase = 0
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('purchase hr and ndcg @%d : %f, %f' %
              (topk[i], hr_purchase, ng_purchase))

        if topk[i] == 20 and total_step >= 10:
            model.best_ckpt_tracker.update_values(total_reward[i], hr_purchase,
                                                  ng_purchase, total_step)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            best_ckpt = model.best_ckpt_tracker.get_max_values()
            print('Best Ckpt Iter: ', (best_ckpt['checkpoint']))
            print('Purchase HR and NDCG @20 %f, %f' % (
                best_ckpt['max_hr_20'], best_ckpt['max_ndcg_20']))
    print('#############################################################')


class SASRecnetwork(torch.nn.Module):

    def __init__(self, hidden_size, learning_rate, item_num, state_size,
                 batch_size, device):
        super(SASRecnetwork, self).__init__()
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.batch_size = batch_size
        self.is_training = torch.BoolTensor()
        self.dev = device

        self.state_embeddings = torch.nn.Embedding(self.item_num + 1,
                                                   self.hidden_size)
        # Positional Encoding
        self.pos_embeddings = torch.nn.Embedding(self.state_size,
                                                 self.hidden_size)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # to be Q for self-attention
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)

        # Build Blocks
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = utils.MultiheadAttention(self.hidden_size,
                                                args.dropout_rate,
                                                num_heads=args.num_heads,
                                                device=self.dev,
                                                causality=True)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = utils.PointWiseFeedForward(self.hidden_size,
                                                 args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        self.cls_layer = torch.nn.Linear(self.hidden_size, self.item_num)

        # The input is expected to contain the unnormalized logits for each class
        # We want the equivalent of tf.nn.sparse_softmax_cross_entropy_with_logits
        self.loss = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.parameters(),
                                    lr=args.lr,
                                    betas=(0.9, 0.999))

        self.best_ckpt_tracker = utils.Tracker()

    def forward(self, state_seq, len_state):
        seqs = self.state_embeddings(torch.LongTensor(state_seq).to(self.dev))
        seqs *= self.state_embeddings.embedding_dim**0.5
        positions = np.tile(np.array(range(state_seq.shape[1])),
                            [state_seq.shape[0], 1])
        seqs += self.pos_embeddings(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = ~torch.BoolTensor(state_seq == self.item_num).to(
            self.dev)
        # broadcast in last dim
        seqs *= timeline_mask.unsqueeze(-1)

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs)
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= timeline_mask.unsqueeze(-1)

        # (U, T, C) -> (U, -1, C)
        seqs = self.last_layernorm(seqs)

        layer_slices = []
        for b_index, len_s in enumerate(len_state - 1):
            last_layer_norm_slice = seqs[b_index, len_s, :]
            layer_slices.append(last_layer_norm_slice)

        logits = torch.stack(layer_slices)

        # The final output is fed to a fully connected layer with no activation func.
        # We use the CrossEntropyLoss which combines a LogSoftmax and NLLLoss.
        return logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Network parameters
    args = parse_args()

    dataset = args.dataset

    if dataset == 'movielens':
        data_directory = 'data/non-reward/movie_lens'
        data_directory += '_' + str(args.seq)
        projname = 'MovieLens'

    elif dataset == 'amazonfood':
        data_directory = 'data/non-reward/amazon_food'
        data_directory += '_' + str(args.seq)
        projname = 'AmazonFood'
    else:
        raise ValueError('Invalid dataset.')

    data_statis = pd.read_pickle(os.path.join(data_directory,
                                              'data_statis.df'))
    state_size = data_statis['state_size'][0]
    item_num = data_statis['item_num'][0]
    pad_item = item_num

    topk = [5, 10, 20]

    print('#############################################################')
    print('Training on dataset : ', projname, 'with sequence length ',
          state_size)
    print('Experiment ID', args.exp_id)
    print('Hyperparams: ')
    print('##############################################')
    print('Batch_size: ', args.batch_size)
    print('Hidden_size: ', args.hidden_factor)
    print('Learning Rate: ', args.lr)
    print('##############################################')

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_interval = args.eval_interval
    # InfoNCE, InstanceInfoNCE
    contrastive_objective = args.contrastive_loss
    console = args.console
    augmentation = args.aug

    SASRec = SASRecnetwork(hidden_size=args.hidden_factor,
                           learning_rate=args.lr,
                           item_num=item_num,
                           state_size=state_size,
                           batch_size=args.batch_size,
                           device=device)

    SASRec = SASRec.to(device)

    replay_buffer = pd.read_pickle(
        os.path.join(data_directory, 'train_replay_buffer.df'))

    eval_buffer = pd.read_pickle(os.path.join(data_directory,
                                              'eval_buffer.df'))

    total_step = 0
    num_rows = replay_buffer.shape[0]
    num_batches = int(num_rows / args.batch_size)
    model_parameters = filter(lambda p: p.requires_grad, SASRec.parameters())
    total_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print('Total number of parameters : ', total_parameters)

    print('Initial Evaluation.')
    evaluate(SASRec, replay_buffer, total_step)
    for i in range(args.epoch):
        for j in range(num_batches):
            batch = replay_buffer.sample(n=args.batch_size).to_dict()
            state = list(batch['state'].values())
            len_state = list(batch['len_state'].values())
            target = list(batch['action'].values())

            state = np.asarray(state)
            len_state = np.asarray(len_state)
            action_target = torch.Tensor(np.asarray(target)).long().to(
                SASRec.dev)
            logits = SASRec(state, len_state)
            prediction = SASRec.cls_layer(logits)

            if augmentation == 'permutation':
                rng = np.random.default_rng()
                states_permuted = rng.permuted(state, axis=1)
                logits_aug_state = SASRec(states_permuted, len_state)
            elif augmentation == 'crop':
                cropped_seq = utils.mask_crop_2d_array(state, pad_item)
                logits_aug_state = SASRec(cropped_seq, len_state)
            elif augmentation == 'mask':
                masked_seq = utils.mask_2d_array(state, pad_item)
                logits_aug_state = SASRec(masked_seq, len_state)

            else:
                raise ValueError('Invalid augmentation.')

            if contrastive_objective == "InfoNCECosine":
                contrastive_loss = utils.info_nce_cosim_loss(logits, logits_aug_state,
                                                 total_step, console)
            elif contrastive_objective == "InfoNCE":
                model_current_state = logits.unsqueeze(-1)
                model_permuted_state = logits_aug_state.unsqueeze(-1)
                contrastive_loss = utils.contrastive_loss(
                    model_current_state, model_permuted_state)

            SASRec.opt.zero_grad()

            loss = SASRec.loss(prediction, action_target)
            final_loss = loss + contrastive_loss
            final_loss.backward()
            SASRec.opt.step()

            total_step += 1

            if total_step % console == 0:
                print("the loss in %dth batch is: %f" % (total_step, loss))
            if total_step % eval_interval == 0:
                evaluate(SASRec, eval_buffer, total_step)
