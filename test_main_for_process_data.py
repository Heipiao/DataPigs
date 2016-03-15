#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-01 22:03:40
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os
import numpy as np
from main_for_process_data import load_data_for_solve
# data = np.array([['1','2.0','3.0','4','5'], ['6','7','32432','9','6'], ['2','3','123','51269','6234']])

# print(data)
# pos = np.where(data[:, 2] == "32432")
# print(pos)
# print(pos[0][0])

# ff = ""
# l = ff[0, 0]
# if data:
# 	print("dassd")
data, features, label = load_data_for_solve("PPD_Master_GBK_2_Test_Set.csv", for_train = False)

print(data.shape)
print(features.shape)
print(features)