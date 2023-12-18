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
    parser = argparse.ArgumentParser(description="SASRec.")

    parser.add_argument('--epoch',
                        type=int,
                        default=1000,
                        help='Number of max epochs.')
    parser.add_argument('--dataset',
                        nargs='?',
                        default='movielens',
                        help='dataset')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size.')
    parser.add_argument('--maxlen', default=10, type=int)
    parser.add_argument('--hidden_factor',
                        type=int,
                        default=32,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--r_click',
                        type=float,
                        default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy',
                        type=float,
                        default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Learning rate.')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_interval', default=2000, type=int)
    parser.add_argument('--exp_id', type=str, default='SASRec_70_30')
    parser.add_argument('--seq', type=int, default=30)
    return parser.parse_args()

def calculate_hit_norew(sorted_list,topk,true_items):

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

def evaluate(model, replay_buffer):
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
        hit_purchase, ndcg_purchase = calculate_hit_norew(sorted_list, topk, actions)
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
            ng_purchase=0
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('purchase hr and ndcg @%d : %f, %f' %
              (topk[i], hr_purchase, ng_purchase))
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

        indices = len_state - 1
        # Create a tensor containing the batch indices
        batch_indices = torch.arange(seqs.size(0))
        # Gather elements using advanced indexing
        gathered = seqs[batch_indices, indices]

        # The final output is fed to a fully connected layer with no activation func.
        # We use the CrossEntropyLoss which combines a LogSoftmax and NLLLoss.
        # output = self.cls_layer(gathered)
        return gathered


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
    reward_click = args.r_click
    reward_buy = args.r_buy
    topk = [5, 10, 20]

    print('#############################################################')
    print('Training on dataset : ', projname, 'with sequence length ', state_size)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_interval = args.eval_interval

    SASRec = SASRecnetwork(hidden_size=args.hidden_factor,
                           learning_rate=args.lr,
                           item_num=item_num,
                           state_size=state_size,
                           batch_size=args.batch_size,
                           device=device)

    SASRec = SASRec.to(device)


    replay_buffer = pd.read_pickle(
        os.path.join(data_directory, 'train_replay_buffer.df'))

    eval_buffer = pd.read_pickle(
        os.path.join(data_directory, 'eval_buffer.df'))

    total_step = 0
    num_rows = replay_buffer.shape[0]
    num_batches = int(num_rows / args.batch_size)
    model_parameters = filter(lambda p: p.requires_grad, SASRec.parameters())
    total_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print('Total number of parameters : ', total_parameters)

    print('Initial Evaluation.')
    evaluate(SASRec, replay_buffer)
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

            SASRec.opt.zero_grad()

            loss = SASRec.loss(prediction, action_target)
            loss.backward()
            SASRec.opt.step()

            total_step += 1

            if total_step % 200 == 0:
                print("the loss in %dth batch is: %f" % (total_step, loss))
            if total_step % eval_interval == 0:
                evaluate(SASRec, eval_buffer)
