#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nltk')
get_ipython().system('pip install wordcloud')
get_ipython().system('pip install snscrape')


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import re
from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import snscrape.modules.twitter as sntwitter
import nltk

nltk.download('vader_lexicon') #required for Sentiment Analysis


# In[2]:


#Get user input
query = input("Query: ")

#As long as the query is valid (not empty or equal to '#')...
if query != '':
    noOfTweet = 2000
    if noOfTweet != '' :
        noOfDays = 180
        if noOfDays != '':
                #Creating list to append tweet data
                tweets_list = []
                now = dt.date.today()
                now = now.strftime('%Y-%m-%d')
                yesterday = dt.date.today() - dt.timedelta(days = int(noOfDays))
                yesterday = yesterday.strftime('%Y-%m-%d')
                for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query + ' lang:en since:' +  yesterday + ' until:' + now + ' -filter:links -filter:replies').get_items()):
                    if i > int(noOfTweet):
                        break
                    tweets_list.append([tweet.date, tweet.content])

                #Creating a dataframe from the tweets list above 
                df = pd.DataFrame(tweets_list, columns=['Datetime', 'Text'])

                


# In[3]:


df


# In[4]:


#Extracting Dates from the datetime field
df['Date'] =pd.to_datetime(df['Datetime']).dt.date


# In[18]:


#del df['Datetime']
df.info()


# In[6]:


input_file = 'DrReddys_Output (1).xlsx'
input_sheet = pd.ExcelFile(input_file)
df_mc = input_sheet.parse('Sheet1')
df_mc.head()


# In[7]:


del df_mc['Unnamed: 0']
del df_mc['0_y']
df_mc.rename(columns = {'News_Headline':'Text'}, inplace = True)
df_mc.head()


# In[20]:


df_mc['Date'] =pd.to_datetime(df_mc['Date']).dt.date


# In[22]:


df_merge = pd.concat([df, df_mc])
df_merge


# In[23]:


# Create a function to clean the tweets
def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
    text = re.sub('#', '', text) # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text) # Removing RT
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
    return text

#applying this function to Text column of our dataframe
df_merge["Text"] = df_merge["Text"].apply(cleanTxt)


# In[24]:


#Adding a new column for Volume/Count of news/Tweets
df_merge['Count'] = pd.Series([1 for x in range(len(df_merge.index))])

#Aggregating the similar date with count and news
d = {'Date': 'first', 'Text': ', '.join, 'Count': 'sum'}
df_new = df_merge.groupby(df_merge['Date'], as_index=False).aggregate(d).reindex(columns=df_merge.columns)
print (df_new)


# In[30]:


#Sentiment Analysis
temp_df = []
for i in range(0,150):
    text = df_new['Text'][i]
    analyzer = SentimentIntensityAnalyzer().polarity_scores(text)
    neg = analyzer['neg']
    neu = analyzer['neu']
    pos = analyzer['pos']
    comp = analyzer['compound']

    if ((pos)==0):
        #negative_list.append(text) #appending the tweet that satisfies this condition
        Positive_words = 0
    else:
        Positive_words = (pos)
        
        #negative += 1 #increasing the count by 1
    if ((neg)==0):
        #positive_list.append(text) #appending the tweet that satisfies this condition
        #positive += 1 #increasing the count by 1
        Negative_words = 0
    else:
        Negative_words = (neg)
        
    if pos == neg:
        #neutral_list.append(text) #appending the tweet that satisfies this condition
        #neutral += 1 #increasing the count by 1      
        Neutral_words = (pos)
    else:
        Neutral_words = 0
    
    temp_df.insert(i,[Positive_words, Negative_words, Neutral_words])

df_t = pd.DataFrame(temp_df, columns = ['Positive_words', 'Negative_words', 'Neutral_words'])
df_t
    
#df_new['Positive'] = Positive_words
#df_new['Negative'] = Negative_words
#df_new['Neutral'] = Neutral_words
#df_new


# In[31]:


#Polarity formula:
#(No. of positive sentiments - No. of negative sentiments)/(Pos+Neg+Neutral)
df_t['Polarity'] = (df_t['Positive_words']-df_t['Negative_words'])/ (df_t['Positive_words']+df_t['Negative_words']+df_t['Neutral_words'])
df_t['Polarity'] = df_t['Polarity'].replace(np.nan, 0)
df_t


# In[32]:


df_final= df_new.join(df_t)


# In[36]:


del df_final['Text']
del df_final['Positive_words']
del df_final['Negative_words']
del df_final['Neutral_words']
df_final.rename(columns = {'Count':'News_Tweet_Volume'}, inplace = True)
df_final


# In[34]:


df_final.to_excel('FinalOutput.xlsx', index = False)

