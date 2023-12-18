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

import os
import pandas as pd
from utility import to_pickled_df, pad_history

if __name__ == '__main__':

    data_directory = 'Retailrocket/'

    length = 20

    sorted_events = pd.read_pickle(
        os.path.join(data_directory, 'sorted_events.df'))
    item_ids = sorted_events.item_id.unique()
    pad_item = len(item_ids)

    train_sessions = pd.read_pickle(
        os.path.join(data_directory, 'sampled_train.df'))
    groups = train_sessions.groupby('session_id')
    ids = train_sessions.session_id.unique()

    state, len_state, action, is_buy, next_state, len_next_state, is_done = [], [], [], [], [],[],[]

    for id in ids:
        group = groups.get_group(id)
        history = []
        for index, row in group.iterrows():
            s = list(history)
            len_state.append(length if len(s) >= length else 1 if len(s) ==
                             0 else len(s))
            s = pad_history(s, length, pad_item)
            a = row['item_id']
            is_b = row['is_buy']
            state.append(s)
            action.append(a)
            is_buy.append(is_b)
            history.append(row['item_id'])
            next_s = list(history)
            len_next_state.append(length if len(next_s) >= length else
                                  1 if len(next_s) == 0 else len(next_s))
            next_s = pad_history(next_s, length, pad_item)
            next_state.append(next_s)
            is_done.append(False)
        is_done[-1] = True

    dic = {
        'state': state,
        'len_state': len_state,
        'action': action,
        'is_buy': is_buy,
        'next_state': next_state,
        'len_next_states': len_next_state,
        'is_done': is_done
    }
    reply_buffer = pd.DataFrame(data=dic)
    to_pickled_df(data_directory, replay_buffer=reply_buffer)

    dic = {'state_size': [length], 'item_num': [pad_item]}
    data_statis = pd.DataFrame(data=dic)
    to_pickled_df(data_directory, data_statis=data_statis)
