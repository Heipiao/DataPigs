#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-02 22:50:18
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from save_load_result import load_result


contents = load_result("after_Str_features_digited_data.csv")
features = np.array(contents[0])
data = np.array(contents[1:])
print(data[0])
print(features[27])

x = np.array(range(1, 30001))
y = data[:, 27]

print(x.shape)
print(y.shape)
plt.scatter(x, y)


plt.savefig("test.png")