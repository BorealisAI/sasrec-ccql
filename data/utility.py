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
import numpy as np


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))


def pad_history(itemlist, length, pad_item):
    if len(itemlist) >= length:
        return itemlist[-length:]
    if len(itemlist) < length:
        temp = [pad_item] * (length - len(itemlist))
        itemlist.extend(temp)
        return itemlist
