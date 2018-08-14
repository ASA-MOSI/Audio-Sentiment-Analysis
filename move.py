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
import shutil

root_path = '/Users/chenfeiyang/PycharmProjects/rename'

for filename in os.listdir('/Users/chenfeiyang/PycharmProjects/rename/data'):

    print(filename),       # 输出每个文件夹的名字，逗号使输出不换行
    print(" : "),
    path = os.path.join('/Users/chenfeiyang/PycharmProjects/rename/data', filename)
    ls = os.listdir(path)
    for i in ls:
        if os.path.isfile(os.path.join(path, i)):
            old = i
            print(old)
            print(root_path + '/data2')
            shutil.move(path+"/"+old, root_path + '/data2')

print("--------")
print("END")
