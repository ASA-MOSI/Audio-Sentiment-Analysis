# -*- coding:utf-8 -*- 
__Author__ = "Feiyang Chen"
__Time__ = '2018/8/14'
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

import numpy as np
import librosa
import math
import re
import pandas as pd


class Extract_feature:

    hop_length = None
    genre_list = ['negative', 'positive', 'neutral']

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
        # self.timeseries = []
        # self.timeseriesindex = []
        # self.timeseriesdict = dict(zip(self.timeseriesindex, self.timeseries))

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

        self.precompute_min_timeseries_len(all_files_list)
        print("[DEBUG] total number of files: " + str(len(self.timeseries_length_list)))

        # Training set
        # print("????????????????????")
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
            # self.timeseriesindex.append(str(file))
            print("Loading " + str(file))
            y, sr = librosa.load(file)
            self.timeseries_length_list.append(math.ceil(len(y) / self.hop_length))
            # self.timeseries.append(math.ceil(len(y) / self.hop_length))
            # self.timeseriesdict = dict(zip(self.timeseriesindex, self.timeseries))
            # print(self.timeseriesdict)

    def extract_audio_features(self, list_of_audiofiles):
        timeseries_length_min = min(self.timeseries_length_list)
        timeseries_length_max = max(self.timeseries_length_list)
        print(timeseries_length_min)
        print(timeseries_length_max)
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
        c = 0

        for l in target:
            if l == "negative":
                a += 1
            elif l == "neutral":
                b += 1
            else:
                c += 1
        print("negative: %d, neutral: %d, positive: %d" % (a, b, c))

        for i, file in enumerate(list_of_audiofiles):
            print(i)
            print(file)
            timeseries_length2 = 256
            # print(timeseries_length2)
            data = np.zeros((len(list_of_audiofiles), timeseries_length2, 33), dtype=np.float64)
            y, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.hop_length, n_mfcc=13)
            mfcc = self.Extended_array(mfcc)
            # print(mfcc.shape)

            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
            spectral_center = self.Extended_array(spectral_center)

            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            chroma = self.Extended_array(chroma)

            # rmse = librosa.feature.rmse(y=y, hop_length=self.hop_length)
            # rmse = self.Extended_array(rmse)
            #
            # chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
            # chroma_cqt = self.Extended_array(chroma_cqt)
            #
            # chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=self.hop_length)
            # chroma_cens = self.Extended_array(chroma_cens)
            #
            # spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)
            # spectral_bandwidth = self.Extended_array(spectral_bandwidth)

            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
            spectral_contrast = self.Extended_array(spectral_contrast)

            # spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)
            # spectral_flatness = self.Extended_array(spectral_flatness)
            #
            # spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)
            # spectral_rolloff = self.Extended_array(spectral_rolloff)
            #
            # poly_features = librosa.feature.poly_features(y=y, sr=sr, hop_length=self.hop_length)
            # poly_features = self.Extended_array(poly_features)
            #
            # tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            # tonnetz = self.Extended_array(tonnetz)
            #
            # zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)
            # zero_crossing_rate = self.Extended_array(zero_crossing_rate)
            #
            # melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=self.hop_length)
            # melspectrogram = self.Extended_array(melspectrogram)

            print(mfcc.T.shape)

            data[i, :, 0:13] = mfcc.T[0:timeseries_length2, :]
            data[i, :, 13:14] = spectral_center.T[0:timeseries_length2, :]
            data[i, :, 14:26] = chroma.T[0:timeseries_length2, :]
            # data[i, :, 26:33] = rmse.T[0:timeseries_length2, :]
            # data[i, :, 33:45] = chroma_cqt.T[0:timeseries_length2, :]
            # data[i, :, 45:57] = chroma_cens.T[0:timeseries_length2, :]
            # # data[i, :, 57:69] = melspectrogram.T[0:timeseries_length2, :]
            # data[i, :, 57:58] = spectral_bandwidth.T[0:timeseries_length2, :]
            # data[i, :, 58:65] = spectral_contrast.T[0:timeseries_length2, :]
            # data[i, :, 65:66] = spectral_flatness.T[0:timeseries_length2, :]
            # data[i, :, 66:67] = spectral_rolloff.T[0:timeseries_length2, :]
            # data[i, :, 67:69] = poly_features.T[0:timeseries_length2, :]
            # data[i, :, 69:75] = tonnetz.T[0:timeseries_length2, :]
            # data[i, :, 75:76] = zero_crossing_rate.T[0:timeseries_length2, :]
            # data[i, :, 76:204] = melspectrogram.T[0:timeseries_length2, :]
            data[i, :, 26:33] = spectral_contrast.T[0:timeseries_length2, :]

            data_path = file

            print("Extracted features audio track %i of %i." % (i + 1, len(list_of_audiofiles)))

        print("data_path:%s " % data_path)
        splits = re.split('[ .]', data_path)
        train_or_test = re.split('[ /]', splits[1])[2]
        print("train_or_test:%s "%train_or_test)
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
            for i in range(1617, 2200):
                file_name = "%s%s%s%s" % (dir_folder, "/", str(i), ".wav")
                list_of_audio.append(file_name)
        return list_of_audio

    def load_target(self, target):
        all_label = pd.read_csv("./OpinionLevelSentiment.csv", header=None)
        arr = np.array(all_label)
        all_label_list = arr.tolist()
        for i in range(0, 2199):
            if all_label_list[i][4] > 1.5:
                target.append("positive")
            elif all_label_list[i][4] < -1.5:
                target.append("negative")
            else:
                target.append("neutral")
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