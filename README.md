# Audio-Sentiment-Analysis
Audio Sentiment Analysis Based on MOSI Dataset

# Data Preprocessing
 - [classify.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/classify.py)
 - [rename.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/rename.py)
 - [move.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/move.py)
 - [counting.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/counting.py)
 
 # Extract Feature && Model && Experiment Results
 ### 2-Class
 - [Extract_feature_2_class.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/Extract_feature_2_class.py)
 - [lstm_keras_2-class.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/lstm_keras_2-class.py)
 
polarity:1454, neutral:745

![Figure 1](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/raw/master/images/2_class_model.png)

metrics| Train | Dev | Test | 
  --- |--- | --- | --- | 
 acc | 0.65 | 0.69 | 0.69 |
 loss | 0.65 | 0.62 | 0.62 |
 

# Plot acc-loss demo
  - [plot_acc-loss.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/plot_acc-loss.py)
 
