# Audio-Sentiment-Analysis
Audio Sentiment Analysis Based on MOSI Dataset

# Data Preprocessing
### Undivided Video
 - [rename0.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/rename0.py)
 
### Divided Video
 - [classify.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/classify.py)
 - [rename.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/rename.py)
 - [move.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/move.py)
 - [counting.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/counting.py)
 
 # Extract Feature && Baseline Model && Experiment Results
 ### 2-Class
 - [Extract_feature_2_class.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/Extract_feature_2_class.py)
 - [lstm_keras_2-class.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/lstm_keras_2-class.py)
 
polarity: 1454, neutral: 745

![Figure 1](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/raw/master/images/2_class_model.png)

metrics| Train | Dev | Test | 
  --- |--- | --- | --- | 
 acc | 0.65 | 0.69 | 0.69 |
 loss | 0.65 | 0.62 | 0.62 |

### 3-Class
 - [Extract_feature_3_class.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/Extract_feature_3_class.py)
 - [lstm_keras_3-class.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/lstm_keras_3-class.py)

negative: 455, neutral: 1199, positive: 545

![Figure 2](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/raw/master/images/3_class_model.png)

metrics| Train | Dev | Test | 
  --- |--- | --- | --- | 
 acc | 0.56 | 0.50 | 0.50 |
 loss | 0.98 | 1.07 | 1.07 |

### 5-Class
 - [Extract_feature_5_class.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/Extract_feature_5_class.py)
 - [lstm_keras_5-class.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/lstm_keras_5-class.py)

negative: 1192, neutral: 103, positive: 482, strong_negative: 185, strong_positive: 237

![Figure 3](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/raw/master/images/5_class_model.png)

metrics| Train | Dev | Test | 
  --- |--- | --- | --- | 
 acc | 0.54 | 0.53 | 0.53 |
 loss | 1.23 | 1.33 | 1.33 |

# Plot acc-loss demo
  - [plot_acc-loss.py](https://github.com/ASA-MOSI/Audio-Sentiment-Analysis/blob/master/plot_acc-loss.py)
 
