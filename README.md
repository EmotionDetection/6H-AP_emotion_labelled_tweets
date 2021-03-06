# Big data for specific emotion detection model: any position and representative emotion hashtag approach  

SanghyubJohn Lee, JongYoon Lim, Leo Paas, HoSeok Ahn

Abstract:
Prediction of human emotion has been and remains a major challenge in many research fields such as psychology, neuroscience, and computer science. Tweets are considered as a suitable source for collecting big data using emotion hashtags as automated emotion annotations. However, little is known about data collection criteria. To elucidate unclear criteria, this paper collected over five million tweets (n=5,645,139) that were divided into six datasets. Machine learning (ML) models were evaluated on both internal (30 analyses) and external test sets (30 analyses), proposing the high-quality emotion labelled dataset (n=1,478,116; any position of representative emotions hashtags). Furthermore, this paper compared the model trained on the proposed dataset with the model trained on a small dataset. We find that this large dataset further improved the model performance in deep learning (18 analyses) than in traditional ML algorithms (30 analyses). Finally, we share the proposed dataset with other researchers to contribute to future specific emotion detection model studies.

Keywords: big data, specific emotion, emotion hashtags, emotion labelled tweets, natural language processing


# Python code example
import pandas as pd

df = pd.read_csv('6H-AP_emotion-labelled_tweets_dataset.dat', sep='\t', encoding='utf-16')

![image](https://user-images.githubusercontent.com/85970005/129505240-13081633-8342-41bb-b86b-e109673090fc.png)

import seaborn as sns

sns.countplot(df['emotions'])

![image](https://user-images.githubusercontent.com/85970005/129505364-f358bc39-137d-4c6d-9094-39ffc908d5f1.png)






