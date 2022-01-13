#!/usr/bin/env python
# coding: utf-8

# In[44]:


from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import openpyxl
import os 
import requests
from lxml import html


# #### Creating Seed url

# In[45]:


seed_url = "https://www.moneycontrol.com"


# In[46]:


input_file = 'Dr_Reddys_Input.xlsx'
input_sheet = pd.ExcelFile(input_file)
df1 = input_sheet.parse('Sheet1')
df1.head()


# #### Creating extraction URL for Dr Reddys stock News

# In[47]:


df1['News_Url']=seed_url+'/company-article/'+df1['Stock_Name']+'/news/'+df1['Stock_Id']


# In[48]:


df1


# #### News Headlines data scraping

# In[49]:


headline=[]
h_date=[]
for i in df1['News_Url']:
    page=requests.get(i)
    if (page.status_code==200):
        tree = html.fromstring(page.content)
        headline_path = (tree.xpath('//*[@class="g_14bl"]/strong/text()'))
        headline.append(headline_path)
        headline_date = (tree.xpath('//p[@class="PT3 a_10dgry"]/text()'))
        headline_date_new=[j.split('\xa0|\xa0', 1)[0] for j in headline_date]
        h_date.append(headline_date_new)
    else:
        headline_path=''
        headline_date=''
        headline.append(headline_path)
        h_date.append(headline_date)


# In[50]:


import itertools


# In[51]:


news_headlines_merged = list(itertools.chain(*headline))


# In[53]:


News_Headlines_df = pd.DataFrame(news_headlines_merged)


# In[54]:


News_Headlines_df


# In[55]:


news_headlines_date_merged = list(itertools.chain(*h_date))


# In[56]:


News_Headlines_date_df = pd.DataFrame(news_headlines_date_merged)


# In[57]:


News_Headlines_date_df


# In[58]:


df2=pd.merge(News_Headlines_df, News_Headlines_date_df, left_index=True, right_index=True)


# #### News Headlines scraped first page

# In[59]:


df2


# #### News Headlines data scraping

# In[60]:


headline1=[]
h_date1=[]
page1=requests.get("https://www.moneycontrol.com/stocks/company_info/stock_news.php?sc_id=DRL&scat=&pageno=2&next=0&durationType=M&Year=&duration=6&news_type=")
if (page1.status_code==200):
    tree1 = html.fromstring(page1.content)
    headline_path1 = (tree1.xpath('//*[@class="g_14bl"]/strong/text()'))
    headline1.append(headline_path1)
    headline_date1 = (tree1.xpath('//p[@class="PT3 a_10dgry"]/text()'))
    headline_date_new1=[j1.split('\xa0|\xa0', 1)[0] for j1 in headline_date1]
    h_date1.append(headline_date_new1)
else:
    headline_path1=''
    headline_date1=''
    headline1.append(headline_path1)
    h_date1.append(headline_date1)


# In[61]:


news_headlines_merged1 = list(itertools.chain(*headline1))


# In[62]:


news_headlines_date_merged1 = list(itertools.chain(*h_date1))


# In[63]:


News_Headlines_df1 = pd.DataFrame(news_headlines_merged1)


# In[64]:


News_Headlines_df1


# In[65]:


News_Headlines_date_df1 = pd.DataFrame(news_headlines_date_merged1)


# In[66]:


News_Headlines_date_df1


# In[67]:


df4=pd.merge(News_Headlines_df1, News_Headlines_date_df1, left_index=True, right_index=True)


# #### News Headlines scraped second page

# In[68]:


df4


# In[71]:


df5=df2.append(df4, ignore_index = True)


# #### News Headlines scraping output for export

# In[76]:


df5


# In[78]:


file_name='DrReddys_Output_v1.xlsx'
df5.to_excel(file_name)

