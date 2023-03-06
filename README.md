# Evaluation of Large Tweet Dataset for Emotion Detection Model: 
# A Comparative Study between Traditional ML and Transformer 
  

SanghyubJohn Lee, JongYoon Lim, Leo Paas, HoSeok Ahn

Abstract:
Specific emotion detection in written human language is a challenging problem in various research fields, including psychology, neuroscience, and computer science. Twitter is a suitable source for collecting a large emotion dataset, as users have provided tweets with emotion hashtags (e.g., #fear, #anger, #sadness, #joy, #surprise, and #disgust) expressing their emotions. However, the criteria for data collection, i.e., the position of representative or synonymous emotion hashtags, remains unclear. In addition, we assess the suitability of various machine learning (ML) algorithms for this purpose. In this study, we collected over five million tweets (n=5,645,139) with 24 emotion hashtags and investigated the efficacy of different criteria for collecting tweets. Contrary to previous research, we found that applying any position of representative emotion hashtags can achieve strong performance, rather than applying the last position of synonymous emotion hashtags. Our study shows that the RoBERTa-large transformer model can significantly improve model performance compared to deep learning algorithms as well as traditional ML algorithms, especially when training on the dataset with a balance between size and quality. We also found that larger datasets are more efficient for RoBERTa model training than smaller datasets when applying the same quality criteria. Finally, we are sharing the proposed emotion dataset, namely, the 6H-AP emotion dataset, with other researchers to advance future specific emotion detection model studies.


# Python code example
import pandas as pd

df = pd.read_csv('6H-AP_emotion-labelled_tweets_dataset.dat', sep='\t', encoding='utf-16')

![image](https://user-images.githubusercontent.com/85970005/129505240-13081633-8342-41bb-b86b-e109673090fc.png)

import seaborn as sns

sns.countplot(df['emotions'])

![image](https://user-images.githubusercontent.com/85970005/129505364-f358bc39-137d-4c6d-9094-39ffc908d5f1.png)






