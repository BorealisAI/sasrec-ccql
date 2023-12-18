# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import torch.nn.functional as F
import random


class PointWiseFeedForward(torch.nn.Module):

    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(
                self.relu(self.dropout1(self.conv1(inputs.transpose(-1,
                                                                    -2))))))
        # as Conv1D requires (N, C, Length)
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class MultiheadAttention(torch.nn.Module):

    def __init__(self,
                 hidden_size,
                 dropout_rate,
                 device,
                 num_heads=8,
                 causality=False):
        super(MultiheadAttention, self).__init__()
        # (N, T_q, C)
        self.Q = torch.nn.Linear(hidden_size, hidden_size)
        self.K = torch.nn.Linear(hidden_size, hidden_size)
        self.V = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.device = device
        self.num_heads = num_heads
        self.causality = causality

    def forward(self, queries, keys):

        Q, K, V = self.Q(queries), self.K(keys), self.V(keys)

        Q_ = Q
        K_ = K
        V_ = V

        # Multiplication
        outputs = Q_.matmul(torch.transpose(K_, 1, 2))

        # Scale
        outputs = outputs / (K_.shape[-1]**0.5)

        # Key Masking
        # (N, T_k)
        key_masks = torch.sign(torch.abs(torch.sum(keys, axis=-1)))

        batch_size = queries.shape[0]
        seq_len = queries.shape[1]
        key_masks = key_masks.tile((seq_len)).reshape(batch_size, seq_len,
                                                      seq_len)

        paddings = torch.ones_like(outputs) * (-2**32 + 1)
        outputs = torch.where((key_masks == 0), paddings, outputs)

        # Causality = Future blinding
        if self.causality:
            # (T_q, T_k)
            diag_vals = torch.ones_like(outputs[0, :, :])
            tril = torch.tril(diag_vals)
            masks = torch.tile(torch.unsqueeze(tril, dim=0),
                               [outputs.shape[0], 1, 1])

            paddings = torch.ones_like(masks) * (-2**32 + 1)
            outputs = torch.where((masks == 0), paddings, outputs)

        # Activation
        # How to deal with softmax inside this function? Doesnt cloning disable  gradients?
        outputs = self.softmax(outputs).clone()

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, axis=-1)))
        query_masks = torch.tile(query_masks.unsqueeze(-1), (1, seq_len))

        outputs *= query_masks

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        # ( h*N, T_q, C/h)
        outputs = torch.matmul(outputs, V_)

        # Residual connection
        outputs += queries

        return outputs


class Tracker:

    def __init__(self):
        self.max_reward = float('-inf')
        self.max_hr_20 = float('-inf')
        self.max_ndcg_20 = float('-inf')
        self.checkpoint = 0

    def update_values(self, reward, hr_20, ndcg_20, checkpoint):
        if reward > self.max_reward and hr_20 > self.max_hr_20 and ndcg_20 > self.max_ndcg_20:
            self.max_reward = reward
            self.max_hr_20 = hr_20
            self.max_ndcg_20 = ndcg_20
            self.checkpoint = max(self.checkpoint, checkpoint)

    def get_max_values(self):
        return {
            'max_reward': self.max_reward,
            'max_hr_20': self.max_hr_20,
            'max_ndcg_20': self.max_ndcg_20,
            'checkpoint': self.checkpoint
        }


def calculate_hit_wqestimates(sorted_list, topk, true_items, rewards, r_click,
                              total_reward, hit_click, ndcg_click,
                              hit_purchase, ndcg_purchase, q_values_neg,
                              q_values_mb, total_qestimates_neg, total_qestimates_mb):

    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    total_qestimates_neg[i] += q_values_neg[j]
                    total_qestimates_mb[i] += q_values_mb[j]
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


def calculate_hit(sorted_list, topk, true_items, rewards, r_click,
                  total_reward, hit_click, ndcg_click, hit_purchase,
                  ndcg_purchase):

    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


def pad_history(itemlist, length, pad_item):
    if len(itemlist) >= length:
        return itemlist[-length:]
    if len(itemlist) < length:
        temp = [pad_item] * (length - len(itemlist))
        itemlist.extend(temp)
        return itemlist


def mask_2d_array(array, pad_item, mask_prob=0.15):
    masked_array = []
    for row in array:
        masked_row = []
        for element in row:
            if random.random() < mask_prob:
                masked_row.append(pad_item)
            else:
                masked_row.append(element)
        masked_array.append(masked_row)
    return np.asarray(masked_array)


def mask_crop_2d_array(array, mask_value=None):
    num_rows, num_cols = array.shape
    masked_array = np.copy(array)
    min_crop_cols, max_crop_cols = 2, 3

    for row in range(num_rows):
        crop_cols = np.random.randint(min_crop_cols, max_crop_cols + 1)
        start_col = np.random.randint(0, num_cols - crop_cols + 1)
        masked_array[row, start_col:start_col + crop_cols] = mask_value

    return masked_array


def contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * contrastive_loss(z1, z2)
        d += 1
    return loss / d


def info_nce_cosim_loss(logitsz1, logitsz2, iteration, console):
    batch_logits = torch.cat([logitsz1, logitsz2])
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(batch_logits[:, None, :],
                                  batch_logits[None, :, :],
                                  dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0],
                          dtype=torch.bool,
                          device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    # InfoNCE loss
    temperature = 0.07
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()
    if iteration % console == 0:
        print('NCE NLL loss ', nll.data.item())
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None],
             cos_sim.masked_fill(pos_mask, -9e15)
             ],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        print("NCE NLL loss _acc_top1",
              (sim_argsort == 0).float().mean().data.item())
        print("NCE NLL loss _acc_top5",
              (sim_argsort < 5).float().mean().data.item())
        print("NCE NLL loss _acc_mean_pos",
              1 + sim_argsort.float().mean().data.item())
    return nll
