import numpy as np
import librosa
import math
import re
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing



class Extract_feature:

    hop_length = None
    genre_list = ['negative', 'neutral', 'positive']

    dir_trainfolder = "./data/_train"

    dir_testfolder = "./data/_test"
    dir_all_files = "./data"

    train_X_preprocessed_data = 'data_train_input.npy'
    train_Y_preprocessed_data = 'data_train_target.npy'

    test_X_preprocessed_data = 'data_test_input.npy'
    test_Y_preprocessed_data = 'data_test_target.npy'

    train_X = train_Y = None

    test_X = test_Y = None

    def __init__(self):
        self.hop_length = 512
        self.timeseries_length_list = []

    def min_max_norm(self,x):
        x_norm = (x - np.mean(x)) / np.std(x)
        return x_norm

    def load_preprocess_data(self):
        self.trainfiles_list = self.path_to_audiofiles(self.dir_trainfolder)

        self.testfiles_list = self.path_to_audiofiles(self.dir_testfolder)

        all_files_list = []
        print("len1:%d"%len(all_files_list))
        all_files_list.extend(self.trainfiles_list)
        print("len2:%d"%len(all_files_list))
        print(all_files_list[0])
        print(all_files_list[1])
        print(all_files_list[2])
        all_files_list.extend(self.testfiles_list)
        print(all_files_list[2198])

    #    self.precompute_min_timeseries_len(all_files_list)
#        print(min(self.timeseries_length_list))
        print("[DEBUG] total number of files: " + str(len(self.timeseries_length_list)))

        # Training set

        print(self.trainfiles_list[0])
        print(self.trainfiles_list[1])
        print(self.trainfiles_list[2])

        self.train_X, self.train_Y = self.extract_audio_features(self.trainfiles_list)
        with open(self.train_X_preprocessed_data, 'wb') as f:
            np.save(f, self.train_X)
        with open(self.train_Y_preprocessed_data, 'wb') as f:
            self.train_Y = self.one_hot(self.train_Y)
            np.save(f, self.train_Y)


        # Test set
        self.test_X, self.test_Y = self.extract_audio_features(self.testfiles_list)
        with open(self.test_X_preprocessed_data, 'wb') as f:
            np.save(f, self.test_X)
        with open(self.test_Y_preprocessed_data, 'wb') as f:
            self.test_Y = self.one_hot(self.test_Y)
            np.save(f, self.test_Y)

    def load_deserialize_data(self):

        self.train_X = np.load(self.train_X_preprocessed_data)
        self.train_Y = np.load(self.train_Y_preprocessed_data)

        self.test_X = np.load(self.test_X_preprocessed_data)
        self.test_Y = np.load(self.test_Y_preprocessed_data)

    def precompute_min_timeseries_len(self, list_of_audiofiles):
        for file in list_of_audiofiles:
            print("Loading " + str(file))
            y, sr = librosa.load(file)
            self.timeseries_length_list.append(math.ceil(len(y) / self.hop_length))


    def extract_audio_features(self, list_of_audiofiles):
        #timeseries_length = min(self.timeseries_length_list)
        print(len(list_of_audiofiles))
        print("*******************")
        print(list_of_audiofiles[0])
        print(list_of_audiofiles[1])
        print(list_of_audiofiles[2])
        print("*******************")
        print(list_of_audiofiles)
        timeseries_length = 8
        data = np.zeros((len(list_of_audiofiles), timeseries_length, 33), dtype=np.float64)
        target = []
        target_ret=[]
        data_path=""

        a=0
        b=0
        c=0
        for l in target:
            if l =="negative":
                a+=1
            elif l=="neutral":
                b+=1
            else:
                c+=1
        print("a:%d,b:%d,c:%d"%(a,b,c))

        for i, file in enumerate(list_of_audiofiles):
            print(i)
            print(file)

            y, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13)
            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)

            print("++++++")
            print(mfcc)
            print(type(mfcc))
            print(mfcc.shape)

            print("{{{{{}}}}}}}")

            data[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
            data[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
            data[i, :, 14:26] = chroma.T[0:timeseries_length, :]
            data[i, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

            data_path=file

            print("Extracted features audio track %i of %i." % (i + 1, len(list_of_audiofiles)))


        print("data_path:%s "%data_path)
        splits = re.split('[ .]', data_path)
        train_or_test = re.split('[ /]', splits[1])[2]
        print("train_or_test:%s "%train_or_test)
        target = self.load_target(target)
        if train_or_test=="_train":
            target_ret[:]=target[0:1616]
        else:
            target_ret[:]=target[1616:2199]

        data=self.min_max_norm(data)

        return data, np.expand_dims(np.asarray(target_ret), axis=1)



    def one_hot(self, Y_genre_strings):
        y_one_hot = np.zeros((Y_genre_strings.shape[0], len(self.genre_list)))
        for i, genre_string in enumerate(Y_genre_strings):
            index = self.genre_list.index(genre_string)
            y_one_hot[i, index] = 1
        return y_one_hot

    def path_to_audiofiles(self, dir_folder):
        list_of_audio = []
        if dir_folder=="./data/_train":
            for i in range(1,1617):
                file_name="%s%s%s%s"%(dir_folder,"/",str(i),".wav")
                list_of_audio.append(file_name)
        else:
            for i in range(1617,2200):
                file_name ="%s%s%s%s"%(dir_folder,"/",str(i),".wav")
                list_of_audio.append(file_name)
        return list_of_audio


    def load_target(self, target):
        all_label = pd.read_csv("./OpinionLevelSentiment.csv", header=None)
        arr = np.array(all_label)
        all_label_list = arr.tolist()
        for i in range(0, 2199):
            if all_label_list[i][4] > 1:
                target.append("positive")
            elif all_label_list[i][4] < -1:
                target.append("negative")
            else:
                target.append("neutral")
        print(len(target))
        return target
