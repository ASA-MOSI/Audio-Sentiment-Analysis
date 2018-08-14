import os

count1 = 0    # 计数大文件夹下共有多少个小文件夹
for filename in os.listdir('/Users/chenfeiyang/PycharmProjects/dataprocessing/data/'):
    print(filename)
    count1 += 1
print(count1)

print("------Print filename done------")

min = 100
max = 0
for filename in os.listdir('/Users/chenfeiyang/PycharmProjects/dataprocessing/data/'):
    print(filename),       # 输出每个文件夹的名字，逗号使输出不换行
    print(" : "),
    count = 0        # 计算小文件夹下音频数量
    path = os.path.join('/Users/chenfeiyang/PycharmProjects/dataprocessing/data/', filename)
    ls = os.listdir(path)
    for i in ls:
        if os.path.isfile(os.path.join(path, i)):
            old = i
            new = i.split("_")[1]
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
