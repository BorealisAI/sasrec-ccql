# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import pandas as pd
import os
import random
import argparse
import time
import utils
import torch.nn.functional as F
import torch.nn.init as init


def parse_args():
    parser = argparse.ArgumentParser(description="SASRec CCQL.")

    parser.add_argument('--epoch',
                        type=int,
                        default=60,
                        help='Number of max epochs.')
    parser.add_argument('--dataset',
                        nargs='?',
                        default='RC15',
                        help='datasets: RC15 or Retailrocket.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor',
                        type=int,
                        default=64,
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
                        default=0.005,
                        help='Learning rate.')
    parser.add_argument('--lr_2',
                        type=float,
                        default=0.001,
                        help='Learning rate.')
    parser.add_argument('--discount',
                        type=float,
                        default=0.5,
                        help='Discount factor for RL.')
    parser.add_argument('--r_negative',
                        type=float,
                        default=-1.0,
                        help='reward for the negative behavior.')
    parser.add_argument('--neg_sample',
                        type=int,
                        default=10,
                        help='number of negative samples.')
    parser.add_argument('--cql_temp',
                        type=float,
                        default=0.5,
                        help='conservativeQTemp.')
    parser.add_argument('--cql_min_q_weight',
                        type=float,
                        default=0.01,
                        help='conservativeQTemp.')
    parser.add_argument('--q_loss_weight',
                        type=float,
                        default=1.0,
                        help='QLossWeight.')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)

    parser.add_argument('--eval_interval', default=2000, type=int)
    parser.add_argument('--switch_interval', default=15000, type=int)
    parser.add_argument('--console', default=500, type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--contrastive_loss',
                        type=str,
                        default='InfoNCECosine')
    parser.add_argument('--aug', type=str, default='permutation')
    parser.add_argument('--exp_id', type=str, default='SASRecCCQL')

    return parser.parse_args()


def evaluate(model, total_step, dataset):
    model.eval()
    eval_sessions = pd.read_pickle(
        os.path.join(data_directory, 'sampled_val.df'))
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated = 0
    total_clicks = 0.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    total_qestimates_neg = [0, 0, 0, 0]
    total_qestimates_mb = [0, 0, 0, 0]
    hit_clicks = [0, 0, 0, 0]
    ndcg_clicks = [0, 0, 0, 0]
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]
    eval_start = time.time()
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        for i in range(batch):
            if dataset == "Retailrocket":
                if evaluated == len(eval_ids):
                    break
            id = eval_ids[evaluated]
            group = groups.get_group(id)
            history = []
            for _, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else
                                  1 if len(state) == 0 else len(state))
                state = utils.pad_history(state, state_size, item_num)
                states.append(state)
                action = row['item_id']
                is_buy = row['is_buy']
                reward = reward_buy if is_buy == 1 else reward_click
                if is_buy == 1:
                    total_purchase += 1.0
                else:
                    total_clicks += 1.0
                actions.append(action)
                rewards.append(reward)
                history.append(row['item_id'])
            evaluated += 1
        states = np.asarray(states)
        len_states = np.asarray(len_states)
        logits = model(states, len_states)
        prediction_tensor = model.output2(logits)
        prediction = prediction_tensor.detach().cpu().numpy()
        sorted_list = np.argsort(prediction)

        # Evaluate Q on negative samples
        # Sample negative actions to evaluate the Q-function.
        all_values = set(range(eval_sessions.item_id.min() + 1, eval_sessions.item_id.max()-1))
        # Remove the values in the input list from the set of all possible values.
        remaining_values = all_values - set(actions)
        # Randomly sample from the remaining values.
        negative_actions = random.sample(remaining_values, len(actions))
        negative_actions = torch.Tensor(np.asarray(negative_actions)).long().to(device)
        q_estimates = model.batched_index(prediction_tensor, negative_actions)
        shifted_q_values = q_estimates + abs(min(q_estimates)) + 1e-8
        shifted_q_values = shifted_q_values.detach().cpu().numpy()

        # Evaluate Q on minibatch
        pos_actions_tensor = torch.Tensor(np.asarray(actions)).long().to(device)
        q_estimates_minibatch = model.batched_index(prediction_tensor, pos_actions_tensor)
        shifted_q_minibatch = q_estimates_minibatch + abs(min(q_estimates_minibatch)) + 1e-8
        shifted_q_minibatch = shifted_q_minibatch.detach().cpu().numpy()

        utils.calculate_hit_wqestimates(sorted_list, topk, actions, rewards,
                                        reward_click, total_reward, hit_clicks,
                                        ndcg_clicks, hit_purchase,
                                        ndcg_purchase, shifted_q_values,
                                        shifted_q_minibatch,
                                        total_qestimates_neg, total_qestimates_mb)
    eval_total_time = time.time() - eval_start
    print('#############################################################')
    print('total clicks: %d, total purchase:%d' %
          (total_clicks, total_purchase))
    for i in range(len(topk)):
        hr_click = hit_clicks[i] / total_clicks
        hr_purchase = hit_purchase[i] / total_purchase
        ng_click = ndcg_clicks[i] / total_clicks
        ng_purchase = ndcg_purchase[i] / total_purchase
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('cumulative reward @ %d: %f' % (topk[i], total_reward[i]))
        print('purchase hr and ndcg @%d : %f, %f' %
              (topk[i], hr_purchase, ng_purchase))
        print('clicks hr ndcg @ %d : %f, %f' % (topk[i], hr_click, ng_click))
        print('q-function estimates neg actions @ %d: %f' % (topk[i], total_qestimates_neg[i]))
        print('q-function estimates mini-batch @ %d: %f' % (topk[i], total_qestimates_mb[i]))
        if topk[i] == 20 and total_step >= 10:
            model.best_ckpt_tracker.update_values(total_reward[i], hr_purchase,
                                                  ng_purchase, total_step)
            print(
                '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            )
            best_ckpt = model.best_ckpt_tracker.get_max_values()
            print('Best Ckpt Iter: ', (best_ckpt['checkpoint']))
            print(
                'Reward: {:.2f}'.format((best_ckpt['max_reward'])),
                'Purchase HR and NDCG @20 %f, %f' %
                (best_ckpt['max_hr_20'], best_ckpt['max_ndcg_20']))
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

        # Initialize the weights of the Embedding layers
        init.normal_(self.state_embeddings.weight, mean=0.0, std=0.01)
        init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.01)

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

        # Initialize the weights of the MultiheadAttention and PointWiseFeedForward layers
        for attn_layer, fwd_layer in zip(self.attention_layers,
                                         self.forward_layers):
            # Initialize weights of MultiheadAttention layer
            init.normal_(attn_layer.Q.weight, mean=0.0, std=0.01)
            init.zeros_(attn_layer.Q.bias)
            init.normal_(attn_layer.K.weight, mean=0.0, std=0.01)
            init.zeros_(attn_layer.K.bias)
            init.normal_(attn_layer.V.weight, mean=0.0, std=0.01)
            init.zeros_(attn_layer.V.bias)
            # Initialize weights of PointWiseFeedForward layer
            init.normal_(fwd_layer.conv1.weight, mean=0.0, std=0.01)
            init.zeros_(fwd_layer.conv1.bias)
            init.normal_(fwd_layer.conv2.weight, mean=0.0, std=0.01)
            init.zeros_(fwd_layer.conv2.bias)

        self.output1 = torch.nn.Linear(self.hidden_size, self.item_num)
        self.output2 = torch.nn.Linear(self.hidden_size, self.item_num)

        # Initialize the weights of the Linear layers
        init.normal_(self.output1.weight, mean=0.0, std=0.01)
        init.zeros_(self.output1.bias)
        init.normal_(self.output2.weight, mean=0.0, std=0.01)
        init.zeros_(self.output2.bias)

        self.celoss1 = torch.nn.CrossEntropyLoss()
        self.celoss2 = torch.nn.CrossEntropyLoss()

        self.opt = torch.optim.Adam(self.parameters(),
                                    lr=args.lr,
                                    betas=(0.9, 0.999))

        self.opt2 = torch.optim.Adam(self.parameters(),
                                     lr=args.lr_2,
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

        model_output = torch.stack(layer_slices)

        return model_output

    def double_qlearning(self, q_vals_state, actions, rewards, discount,
                         q_vals_next_state, q_vals_next_state_selector):
        """ Double-Q operator.
          Args:
            q_vals_state: Tensor holding Q-values for s in a batch of transitions,
                shape `[B x num_actions]`.
            actions: Tensor holding action indices, shape `[B]`.
            rewards: Tensor holding rewards, shape `[B]`.
            discount: Tensor holding pcontinue values, shape `[B]`.
            q_vals_next_state: Tensor of Q-values for s' in a batch of transitions,
                used to estimate the value of the best action, shape `[B x num_actions]`.
            q_vals_next_state_selector: Tensor of Q-values for s' in a batch of
                transitions used to estimate the best action, shape `[B x num_actions]`.
        """

        with torch.no_grad():
            # Build target and select head to update.
            best_action = torch.argmax(q_vals_next_state_selector, 1)
            double_q_bootstrapped = self.batched_index(q_vals_next_state,
                                                       best_action)
            target = rewards + discount * double_q_bootstrapped

        q_pred = self.batched_index(q_vals_state, actions)
        # Temporal difference error and loss.
        # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
        td_error = target - q_pred
        loss = 0.5 * torch.square(td_error)

        return loss, q_pred, best_action

    def batched_index(self, values, indices):
        """Equivalent to `values[:, indices]` or tf.gather`.
        """
        one_hot_indices = torch.nn.functional.one_hot(
            indices, num_classes=self.item_num)
        sum_vals = torch.sum(values * one_hot_indices, dim=-1)
        return sum_vals


class SASRecnetworkConservative():

    def __init__(self, args, device):

        self.qf1 = SASRecnetwork(hidden_size=args.hidden_factor,
                                 learning_rate=args.lr,
                                 item_num=item_num,
                                 state_size=state_size,
                                 batch_size=args.batch_size,
                                 device=device)

        self.qf2 = SASRecnetwork(hidden_size=args.hidden_factor,
                                 learning_rate=args.lr,
                                 item_num=item_num,
                                 state_size=state_size,
                                 batch_size=args.batch_size,
                                 device=device)

        self.qf1 = self.qf1.to(device)
        self.qf2 = self.qf2.to(device)

        self.cql_temp = args.cql_temp
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max = np.inf
        self.cql_min_q_weight = args.cql_min_q_weight
        self.q_loss_weight = args.q_loss_weight

        self.batch_size = args.batch_size

        self.reward_negative = args.r_negative
        self.negative_samples = args.neg_sample

    def contrastive_loss(self, mainQN, current_state_logits, states,
                         len_states):

        if augmentation == 'permutation':
            rng = np.random.default_rng()
            states_permuted = rng.permuted(states, axis=1)
            logits_aug_state = mainQN(states_permuted, len_states)
        elif augmentation == 'crop':
            cropped_seq = utils.mask_crop_2d_array(states, pad_item)
            logits_aug_state = mainQN(cropped_seq, len_states)
        elif augmentation == 'mask':
            masked_seq = utils.mask_2d_array(states, pad_item)
            logits_aug_state = mainQN(masked_seq, len_states)

        else:
            raise ValueError('Invalid augmentation.')

        if contrastive_objective == "InfoNCECosine":
            contrastive_loss = utils.info_nce_cosim_loss(
                current_state_logits, logits_aug_state, total_step, console)
        elif contrastive_objective == "InfoNCE":
            current_state_logits = current_state_logits.unsqueeze(-1)
            logits_aug_state = logits_aug_state.unsqueeze(-1)
            contrastive_loss = utils.contrastive_loss(current_state_logits,
                                                      logits_aug_state)
        elif contrastive_objective == "Hierarchical":
            current_state_logits = current_state_logits.unsqueeze(-1)
            logits_aug_state = logits_aug_state.unsqueeze(-1)
            contrastive_loss = utils.hierarchical_contrastive_loss(
                current_state_logits, logits_aug_state)

        return contrastive_loss

    def cql_loss(self, action, target_Qs, q_tm1, q_pred, discounts,
                 target_Qs_selector):

        neg_reward = torch.full((self.batch_size, ),
                                self.reward_negative).to(device)
        negative_actions = []

        for index in range(target_Qs.shape[0]):
            negative_list = []
            for i in range(self.negative_samples):
                neg = np.random.randint(item_num)
                while neg == action[index]:
                    neg = np.random.randint(item_num)
                negative_list.append(neg)
            negative_actions.append(negative_list)
        negative_actions = torch.Tensor(
            np.asarray(negative_actions)).long().to(device)

        neg_act_predictions = []

        for i in range(self.negative_samples):
            negatives = negative_actions[:, i]
            _, q_pred_neg, action_preds_neg = mainQN.double_qlearning(
                q_tm1, negatives, neg_reward, discounts, target_Qs,
                target_Qs_selector)
            neg_act_predictions.append(action_preds_neg)
        # Feed-in action predictions.
        neg_act_predictions = torch.cat(neg_act_predictions).reshape(
            target_Qs.shape[0], self.negative_samples)
        _, q_pred_neg, cql_q1_current_actions = mainQN.double_qlearning(
            q_tm1, action_preds_neg, neg_reward, discounts, target_Qs,
            target_Qs_selector)

        # Conservative Q-Learning
        q_pred_neg = torch.unsqueeze(q_pred_neg, 1)
        q_pred = torch.unsqueeze(q_pred, 1)
        cql_q1_current_actions = torch.unsqueeze(cql_q1_current_actions, 1)
        cql_cat_q1 = torch.cat([q_pred_neg, q_pred, cql_q1_current_actions],
                               dim=1)

        cql_std = torch.std(cql_cat_q1, dim=1)

        cql_qf_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp,
                                     dim=1) * self.cql_temp
        """Subtract the log likelihood of data"""
        cql_qf_diff = torch.clamp(
            cql_qf_ood - q_pred,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        cql_min_qf_loss = cql_qf_diff * self.cql_min_q_weight

        return cql_min_qf_loss, cql_std

    def train(self, mainQN, target_QN, batch, total_step):

        state = list(batch['state'].values())
        len_state = list(batch['len_state'].values())
        action = list(batch['action'].values())
        is_buy = list(batch['is_buy'].values())

        next_state = list(batch['next_state'].values())
        len_next_state = list(batch['len_next_states'].values())
        next_states = np.asarray(next_state)
        len_next_states = np.asarray(len_next_state)

        reward = []
        for k in range(len(is_buy)):
            reward.append(reward_buy if is_buy[k] == 1 else reward_click)
        discount = [args.discount] * len(action)

        states = np.asarray(state)
        len_states = np.asarray(len_state)
        actions = torch.Tensor(np.asarray(action)).long().to(device)
        rewards = torch.Tensor(np.asarray(reward)).to(device)
        discounts = torch.Tensor(np.asarray(discount)).to(device)

        target_model_output = target_QN(next_states, len_next_states)
        target_Qs = target_QN.output1(target_model_output)

        main_model_output = mainQN(next_states, len_next_states)
        target_Qs_selector = mainQN.output1(main_model_output)

        clip_value = 1.0

        # Set target_Qs to 0 for states where episode ends
        is_done = list(batch['is_done'].values())
        for index in range(target_Qs.shape[0]):
            if is_done[index]:
                target_Qs[index] = torch.Tensor(np.zeros([item_num
                                                          ])).to(device)

        if total_step < switch_interval:
            main_model_current_state = mainQN(states, len_states)
            q_tm1 = mainQN.output1(main_model_current_state)

            q_loss, q_pred, _ = mainQN.double_qlearning(
                q_tm1, actions, rewards, discounts, target_Qs,
                target_Qs_selector)

            predictions = mainQN.output2(main_model_current_state)

            contrastive_loss = self.contrastive_loss(mainQN,
                                                     main_model_current_state,
                                                     states, len_states)

            mainQN.opt.zero_grad()
            loss = mainQN.celoss1(predictions, actions)

            cql_min_qf_loss, cql_std = self.cql_loss(action, target_Qs, q_tm1,
                                                     q_pred, discounts,
                                                     target_Qs_selector)

            final_loss = torch.mean(loss + (self.q_loss_weight*q_loss) + cql_min_qf_loss +
                                    contrastive_loss)
            final_loss.backward()
            for param in mainQN.parameters():
                param.grad.data.clamp_(-clip_value, clip_value)

            mainQN.opt.step()

            if total_step % console == 0:
                print("the loss in %dth batch is: %f" %
                      (total_step, final_loss))

        else:
            main_model_current_state = mainQN(states, len_states)
            q_tm1 = mainQN.output1(main_model_current_state)

            q_loss, q_pred, _ = mainQN.double_qlearning(
                q_tm1, actions, rewards, discounts, target_Qs,
                target_Qs_selector)

            predictions = mainQN.output2(main_model_current_state)

            with torch.no_grad():
                q_indexed = mainQN.batched_index(q_tm1, actions)

            contrastive_loss = self.contrastive_loss(mainQN,
                                                     main_model_current_state,
                                                     states, len_states)

            celoss2 = mainQN.celoss2(predictions, actions)
            loss_multi = torch.multiply(q_indexed, celoss2)

            cql_min_qf_loss, cql_std = self.cql_loss(action, target_Qs, q_tm1,
                                                     q_pred, discounts,
                                                     target_Qs_selector)

            mainQN.opt2.zero_grad()
            final_loss = torch.mean(loss_multi + (self.q_loss_weight*q_loss) + cql_min_qf_loss +
                                    contrastive_loss)
            final_loss.backward()
            for param in mainQN.parameters():
                param.grad.data.clamp_(-clip_value, clip_value)

            mainQN.opt2.step()

            if total_step % console == 0:
                print("the loss in %dth batch is: %f" %
                      (total_step, final_loss))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Network parameters
    args = parse_args()

    dataset = args.dataset

    data_directory = 'data/' + dataset + '/'
    data_statis = pd.read_pickle(os.path.join(data_directory,
                                              'data_statis.df'))
    state_size = data_statis['state_size'][0]
    item_num = data_statis['item_num'][0]
    pad_item = item_num
    reward_click = args.r_click
    reward_buy = args.r_buy
    topk = [5, 10, 20]

    if dataset == "Retailrocket":
        # Overwrite some parameters
        args.contrastive_loss = 'InfoNCE'
        args.batch_size = 512
        args.lr = 0.01
        args.r_negative = -3.0
        args.cql_temp = 0.5
        args.switch_interval = 50000
        args.lr_2 = args.lr

    console = args.console
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    eval_interval = args.eval_interval
    switch_interval = args.switch_interval
    # InfoNCECosine, InfoNCE, Hierarchical
    contrastive_objective = args.contrastive_loss
    augmentation = args.aug

    model = SASRecnetworkConservative(args, device)

    replay_buffer = pd.read_pickle(
        os.path.join(data_directory, 'replay_buffer.df'))

    total_step = 0
    num_rows = replay_buffer.shape[0]
    num_batches = int(num_rows / args.batch_size)
    model_parameters = filter(lambda p: p.requires_grad,
                              model.qf1.parameters())
    total_parameters = sum([np.prod(p.size()) for p in model_parameters])
    print('Total number of parameters : ', total_parameters * 2)
    print('Model : SASRec Contrastive CQL')
    print('Dataset : ', dataset)
    print('Experiment ID', args.exp_id)
    print('Seed: ', args.seed)
    print('Hyperparams: ')
    print('##############################################')
    print('Batch_size: ', args.batch_size)
    print('Hidden_size: ', args.hidden_factor)
    print('Learning Rate: ', args.lr)
    print('Discount: ', args.discount)
    print('Contrastive Loss: ', contrastive_objective)
    print('Augmentation : ', augmentation)
    print('Negative Reward: ', args.r_negative)
    print('Negative Samples: ', args.neg_sample)
    print('CQL Temperature: ', args.cql_temp)
    print('CQL Min Q Weight: ', args.cql_min_q_weight)
    print('Q Loss Weight: ', args.q_loss_weight)
    print('Reward buy: ', args.r_buy)
    print('Reward click: ', args.r_click)
    print('##############################################')

    print('Initial Evaluation.')
    evaluate(model.qf1, total_step, dataset)
    for i in range(args.epoch):
        for j in range(num_batches):
            batch = replay_buffer.sample(n=args.batch_size).to_dict()

            # double q learning, pointer is for selecting which network
            # is target and which is main.
            pointer = np.random.randint(0, 2)
            model.qf1.train()
            model.qf2.train()
            # Instead of using both q's and selecting min
            # we just alternate between the two.
            if pointer == 0:
                mainQN = model.qf1
                target_QN = model.qf2
            else:
                mainQN = model.qf2
                target_QN = model.qf1

            model.train(mainQN, target_QN, batch, total_step)

            total_step += 1
            if total_step % eval_interval == 0:
                evaluate(mainQN, total_step, dataset)
