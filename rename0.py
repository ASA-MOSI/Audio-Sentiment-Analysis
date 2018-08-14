import os
import pandas as pd
import numpy as np

#for file in os.listdir("./data/filerename"):
 #   print(type(file))

target=[]
targetIndex=[]

for i in range(1,2200):
    targetIndex.append(i)

all_label = pd.read_csv("./OpinionLevelSentiment.csv", header=None)
arr = np.array(all_label)
all_label_list = arr.tolist()

for i in range(0, 2199):
    target.append(all_label_list[i][2]+"_"+str(all_label_list[i][3])+".wav")

target_dic=dict(zip(target,targetIndex))
print(target_dic)


path="./data/_train"
for file in os.listdir(path):
    for j in target_dic:
        if(file==j):
            os.rename(path+"/"+file,path+"/"+str(target_dic[j])+".wav")

path="./data/_test"
for file in os.listdir(path):
    for j in target_dic:
        if(file==j):
            os.rename(path+"/"+file,path+"/"+str(target_dic[j])+".wav")

print("END")
