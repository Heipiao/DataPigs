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

data = np.array([['1','2.0','3.0','4','5'], ['6','7','32432','9','6'], ['2','3','123','51269','6234']])

print(data)
pos = np.where(data[:, 2] == "32432")
print(pos)
print(pos[0][0])

ff = ""
l = ff[0, 0]
if data:
	print("dassd")