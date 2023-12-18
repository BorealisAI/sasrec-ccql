# Copyright (c) 2019-present, Royal Bank of Canada.
# Copyright (c) 2019-present, Michael Kelly.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the implementation provided by the authors of https://arxiv.org/pdf/2006.05779.pdf
#  and https://arxiv.org/pdf/2111.03474.pdf: Xin Xin, Alexandros Karatzoglou, Ioannis Arapakis
# Joemon M. Jose.
#################################################################################### 

import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split

def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))

def process_sequence(sequence, n, pad_item):
    seq_length = len(sequence)

    if seq_length > n:

        # Split the sequence into smaller n-length lists
        chunks = [sequence[i:i + n] for i in range(0, seq_length, n)]
        chunk_lengths = [len(chunk) for chunk in chunks]
        assert(sum(chunk_lengths) == seq_length)
        actions = [chunks[i][-1] for i in range(len(chunks))]
        assert(chunks[0][-1] == actions[0])

        for i in range(len(chunks)):
            chunks[i] = chunks[i][:-1]
            chunk_lengths[i] -= 1

        # Pad the last chunk if its length is less than n
        if len(chunks[-1]) < n:
            padding = [pad_item] * (n - len(chunks[-1]) - 1)
            chunks[-1] = chunks[-1] + padding

    else:
        actions = [sequence[-1]]
        sequence.pop()
        seq_length = len(sequence)
        chunk_lengths = [seq_length]
        # Pad the sequence with the pad_value
        padding = [pad_item] * (n - seq_length - 1)
        chunks = [sequence + padding]

    assert(max(chunk_lengths) <= n)

    return chunks, chunk_lengths, actions


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='movielens')
    parser.add_argument('--seq', type=int, default=30)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    args = parser.parse_args()

    # movielens or amazonfood
    dataset = args.data
    sequence_length = args.seq

    print('#############################################################')
    print('Processing dataset : ', dataset, 'with sequence length ', sequence_length)

    # Need to run and store these separately for torch vs tf versions.
    if dataset == 'movielens':
        data_directory = 'non-reward/movie_lens'
        df_file = 'movie_lens.df'
    elif dataset == 'amazonfood':
        data_directory = 'non-reward/amazon_food'
        df_file = 'amazon_food.df'
    else:
        raise ValueError('Dataset not supported.')

    seed = args.seed

    traj_length= sequence_length + 1

    train_sessions = pd.read_pickle(os.path.join(data_directory, df_file))
    groups=train_sessions.groupby('user_id')
    ids=train_sessions.user_id.unique()
    pad_item = train_sessions.item_id.max()+1

    user_state, user_len_state, user_action= [], [], []

    for id in ids:
        group=groups.get_group(id)
        states = group.item_id.tolist()

        states, len_states, actions = process_sequence(states, traj_length, pad_item)

        user_state.extend(states)
        user_len_state.extend(len_states)
        user_action.extend(actions)

    # Set the ratios for the data splits
    train_ratio = args.train_ratio
    test_ratio = round(1.0 - args.train_ratio, 2)

    print('Train Ratio: ', train_ratio)
    print('Test Ratio: ', test_ratio)

    dic={'state':user_state,'len_state':user_len_state,'action':user_action}
    replay_buffer=pd.DataFrame(data=dic)

    # Perform the first split: train and test
    train_data, test_data = train_test_split(replay_buffer, test_size=(test_ratio), random_state=seed)

    print('# Train Interactions: ', len(train_data))
    print('# Test Interactions: ', len(test_data))
    print('#############################################################')

    data_directory += '_' + str(sequence_length)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    to_pickled_df(data_directory, train_replay_buffer=train_data)
    to_pickled_df(data_directory, eval_buffer=test_data)

    dic={'state_size':[traj_length - 1],'item_num':[pad_item]}
    data_statis=pd.DataFrame(data=dic)
    to_pickled_df(data_directory, data_statis=data_statis)
