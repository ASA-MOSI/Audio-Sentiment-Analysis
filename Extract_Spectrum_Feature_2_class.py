# -*- coding:utf-8 -*- 
__Author__ = "Feiyang Chen"
__Time__ = '2018/8/16'
__Title__ = 'Audio_Sentiment_Analysis'

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

from scipy.io import wavfile
from matplotlib import pylab
import cv2
import numpy as np
import librosa
import math
import re
import pandas as pd
import wave





class Extract_Spectrum_feature:

    hop_length = None
    genre_list = ['neutral', 'polarity']

    dir_trainfolder = "./data/_train"
    dir_testfolder = "./data/_test"
    dir_all_files = "./data"

    train_X_preprocessed_data = 'data_train_input2.npy'
    train_Y_preprocessed_data = 'data_train_target2.npy'

    test_X_preprocessed_data = 'data_test_input2.npy'
    test_Y_preprocessed_data = 'data_test_target2.npy'

    train_X = train_Y = None

    test_X = test_Y = None

    def __init__(self):
        self.hop_length = 512
        self.timeseries_length_list = []
        # self.timeseries = []
        # self.timeseriesindex = []
        # self.timeseriesdict = dict(zip(self.timeseriesindex, self.timeseries))

    def spectrogram(self, path):
        sr, d = wavfile.read(path)
        wavefile = wave.open(path, 'r')

        nchannels = wavefile.getnchannels()

        if nchannels == 1:

            print("channels should be 1: %d" % nchannels)
            channels = [
                np.array(d[:])
            ]
            s, f, t, im = pylab.specgram(channels[0], Fs=sr)
            im.figure.gca().set_axis_off()
            return im.figure
        else:
            print("channels should be 2: %d" % nchannels)
            channels = [
                np.array(d[:, 0]),
                np.array(d[:, 1])
            ]
            s, f, t, im = pylab.specgram(channels[1], Fs=sr)
            im.figure.gca().set_axis_off()
            return im.figure

    def load_preprocess_data(self):

        self.trainfiles_list = self.path_to_audiofiles(self.dir_trainfolder)
        self.testfiles_list = self.path_to_audiofiles(self.dir_testfolder)

        all_files_list = []
        # print("长度1是： %d" % len(all_files_list))
        all_files_list.extend(self.trainfiles_list)
        # print("长度2是： %d" % len(all_files_list))
        print(all_files_list[0])
        print(all_files_list[1])
        print(all_files_list[2])
        all_files_list.extend(self.testfiles_list)
        print(all_files_list[2198])

      #  self.precompute_min_timeseries_len(all_files_list)
        print("[DEBUG] total number of files: " + str(len(self.timeseries_length_list)))

        # Training set
        # print("????????????????????")
     #   print(self.trainfiles_list[0])
     #   print(self.trainfiles_list[1])
     #   print(self.trainfiles_list[2])

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
            # self.timeseriesindex.append(str(file))
            print("Loading " + str(file))
            y, sr = librosa.load(file)
            self.timeseries_length_list.append(math.ceil(len(y) / self.hop_length))
            # self.timeseries.append(math.ceil(len(y) / self.hop_length))
            # self.timeseriesdict = dict(zip(self.timeseriesindex, self.timeseries))
            # print(self.timeseriesdict)

    def extract_audio_features(self, list_of_audiofiles):
        #timeseries_length_min = min(self.timeseries_length_list)
        #timeseries_length_max = max(self.timeseries_length_list)
        #print(timeseries_length_min)
        #print(timeseries_length_max)
        print(len(list_of_audiofiles))
        print("*******************")
        print(list_of_audiofiles[0])
        print(list_of_audiofiles[1])
        print(list_of_audiofiles[2])
        print("*******************")
        print(list_of_audiofiles)

        target = []
        target_ret = []
        data_path = ""
        # ./data/_train/dvsfs.wav

        target = self.load_target(target)
        print("+++++++++++")
        print(len(target))
        print("+++++++++++")
        target_ret[:] = target[0:1616]
        print(len(target_ret))
        print(target)
        print("+++++++++++")
        a = 0
        b = 0
        for l in target:
            if l == "polarity":
                a += 1
            elif l == "neutral":
                b += 1
        print("polarity:%d, neutral:%d" % (a, b))

        for i, file in enumerate(list_of_audiofiles):
            print(i)
            print(file)
            timeseries_length2 = 256
            # print(timeseries_length2)
            data = np.zeros((len(list_of_audiofiles), 512, 512, 3), dtype=np.int32)


            self.spectrogram(file).savefig('./images/'+str(i)+'.png', bbox_inches='tight', pad_inches=0)

            print("size after reshape")

            print((cv2.resize(cv2.imread('./images/' + str(i) + '.png', 1), (512, 512),
                              interpolation=cv2.INTER_AREA)).shape)

            data[i, :, :, :] = cv2.resize(cv2.imread('./images/' + str(i) + '.png', 1), (512, 512),
                                          interpolation=cv2.INTER_AREA)
            data_path = file

            print("Extracted features audio track %i of %i." % (i + 1, len(list_of_audiofiles)))

        print("data_path:%s " % data_path)
        splits = re.split('[ .]', data_path)
        train_or_test = re.split('[ /]', splits[1])[2]
        print("train_or_test:%s " % train_or_test)
        target = self.load_target(target)
        if train_or_test == "_train":
            target_ret[:] = target[0:1616]
        else:
            target_ret[:] = target[1616:2199]
        return data, np.expand_dims(np.asarray(target_ret), axis=1)

    def one_hot(self, Y_genre_strings):
        y_one_hot = np.zeros((Y_genre_strings.shape[0], len(self.genre_list)))
        for i, genre_string in enumerate(Y_genre_strings):
            index = self.genre_list.index(genre_string)
            y_one_hot[i, index] = 1
        return y_one_hot

    def path_to_audiofiles(self, dir_folder):
        list_of_audio = []
        if dir_folder == "./data/_train":
            for i in range(1, 1617):
                file_name = "%s%s%s%s" % (dir_folder, "/", str(i), ".wav")
                list_of_audio.append(file_name)
        else:
            for i in range(1617,2200):
                file_name = "%s%s%s%s" % (dir_folder, "/", str(i), ".wav")
                list_of_audio.append(file_name)
        return list_of_audio

    def load_target(self, target):
        all_label = pd.read_csv("./OpinionLevelSentiment.csv", header=None)
        arr = np.array(all_label)
        all_label_list = arr.tolist()
        for i in range(0, 2199):
            if all_label_list[i][4] > -1 and all_label_list[i][4] <= 1:
                target.append("neutral")
            else:
                target.append("polarity")
        # print(len(target))
        return target

    def Extended_array(self, feature):
        if feature.shape[1] < 256:
            # print(feature.shape)
            feature_mean = np.mean(feature, axis=1)
            feature_mean = feature_mean.reshape(len(feature_mean), -1)
            # print(feature_mean.shape)
            feature_extend = np.tile(feature_mean, 256-feature.shape[1])
            # print(feature_extend.shape)
            feature_extended = np.concatenate((feature, feature_extend), axis=1)
            # print(feature_extended)
            return feature_extended
        else:
            return feature