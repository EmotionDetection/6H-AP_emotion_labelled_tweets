# Specific_Emotion

Prediction of human emotion has been and remains a major challenge in many research fields such as psychology, neuroscience, and computer science. Tweets are considered as a suitable source for collecting big data using emotion hashtags as automated emotion annotations. However, little is known about data collection criteria. To elucidate unclear criteria, this paper collected over five million tweets (n=5,626,219) that were divided into six datasets. Machine learning (ML) models were evaluated on both internal (30 analyses) and external test sets (30 analyses), proposing the 6H-AP dataset (n=1,367,254; any position of representative emotions hashtags). Furthermore, this paper cross evaluates the 6H-AP dataset with a small dataset, showing that this large dataset can make a greater contribution to improving model performance in deep learning (12 analyses) than in traditional ML algorithms (20 analyses). Finally, we share the 6H-AP dataset with other researchers to contribute to future specific emotion detection model studies.
Keywords: big data, specific emotion, emotion hashtags, emotion labelled tweets, natural language processing

# Python code example
import pandas as pd
df = pd.read_csv('6H-AP_emotion-labelled_tweets_dataset.dat', sep='\t', encoding='utf-16')


id	tweets	hastags	emotions	length
17	50414935	How i wish we had a competent government No wo...	anger	#anger	104
25	50144143	We must help the children So much So much viol...	anger	#anger	63
29	50341011	Getting woken up now and I have to be up at ahh	anger	#anger	48
52	50079290	Your anger only truly hurts one person and tha...	anger	#anger	241
61	50332223	come out let's get real about anger management...	anger	#anger	86
...	...	...	...	...	...
3516874	55331963	Shoutout to my New Followers I usually Clown ...	surprise	#surprise	96
3516896	55381596	it is friday I'm wearing my fridaynosaur shirt...	surprise	#surprise	65
3516953	55220607	Whoa I didnotexpectthat	surprise	#surprise	24
3516955	55057571	Zero I don t need help at the moment sewing ca...	surprise	#surprise	62
3516962	55001309	Happy Friday to me When you get a surprise gif...	surprise	#surprise	105

