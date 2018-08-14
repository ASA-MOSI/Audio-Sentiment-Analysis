# -*- coding:utf-8 -*- 
__Author__ = "Feiyang Chen"
__Time__ = '2018/8/13'
__Title__ = 'rename'

"""
 code is far away from bugs with the god animal protecting
    I love animals. God bless you.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃ 永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

import os
import pandas as pd
import numpy as np


target = []
targetIndex = []

for i in range(1, 94):
    targetIndex.append(i)

print(targetIndex)

all_label = pd.read_csv("./OpinionLevelSentiment.csv", header=None)
arr = np.array(all_label)
all_label_list = arr.tolist()

target = set(target)
for i in range(0, 2199):
    # print(all_label_list[i][2][0:7])
    target.add(all_label_list[i][2][0:7])

print(target)

target_dic = dict(zip(target, targetIndex))
print(target_dic)


path = "./data"
for file in os.listdir(path):
    for j in target_dic:
        if file == j:
            os.rename(path+"/"+file, path+"/"+str(target_dic[j]))

print("------Dir Rename Done------")

count1 = 0    # 计数大文件夹下共有多少个小文件夹
for filename in os.listdir('/Users/chenfeiyang/PycharmProjects/rename/data'):
    print(filename)
    count1 += 1
print(count1)

print("------Print filename done------")

min = 100
max = 0
for filename in os.listdir('/Users/chenfeiyang/PycharmProjects/rename/data'):
    print(filename),       # 输出每个文件夹的名字，逗号使输出不换行
    print(" : "),
    count = 0        # 计算小文件夹下音频数量
    path = os.path.join('/Users/chenfeiyang/PycharmProjects/rename/data', filename)
    ls = os.listdir(path)
    for i in ls:
        if os.path.isfile(os.path.join(path, i)):
            old = i
            new = i.split("_")[-1]
            print(old)
            print(new)
            print(path)
            print(filename)
            print(path+"/"+old)
            print(path+"/"+filename+"_"+new)
            os.rename(path+"/"+old, path+"/"+filename+"_"+new)
            count += 1
    print(count)
    if count > max:
        max = count
    if count < min:
        min = count
print("--------")
print(max)
print(min)
print("END")
