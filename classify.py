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


'''
    根据目录下的文件名创建所需要的文件夹，已经存在的文件夹跳过。
    '''
def creatDir(names):
    dirList = []
    for name in names:
        if os.path.isdir(name):
            print("%s is a dir." % (name))
            continue
        dirList.append(name[0:7])
    dirList = list(set(dirList))
    print("将要创建的文件夹为：%s" % dirList)
    print("将在 %s 下创建。" % absTargetPath)
    os.chdir(absTargetPath)
    for path in dirList:
        if os.path.exists(path):
            print("文件夹 %s 已经存在!" % path)
        else:
            os.makedirs(path)
            print("创建文件夹 %s 成功！" % path)


'''
    将文件放到对应的文件夹中
    '''
def classify(names):
    for n in range(len(nameList)):
        old = os.path.join(absTargetPath, nameList[n])
        new = os.path.join(absTargetPath, nameList[n][0:7])
        '''
        移动文件
        '''
        os.rename(old, new + '/' + nameList[n])
        print("文件 %s 移动到 %s 中，done！" % (nameList[n], new))


if __name__ == '__main__':
    docPath = '/Users/chenfeiyang/PycharmProjects/dataprocessing/data/'
    print('需要进行分类的目标目录绝对路径为：')
    absTargetPath = os.path.abspath(docPath)
    print(absTargetPath)
    nameList = []
    for name in os.listdir(absTargetPath):
        if os.path.isfile(absTargetPath + '/' + name):
            nameList.append(name)
    print("包含文件 %s 个。" % len(nameList))
    creatDir(nameList)
    classify(nameList)
