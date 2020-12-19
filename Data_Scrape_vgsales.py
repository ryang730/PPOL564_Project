#!/usr/bin/env python
# coding: utf-8

# In[106]:


import requests
import urllib
import pandas as pd
import country_converter as coco
from bs4 import BeautifulSoup #for parsing the website
import time #for system to sleep 
import random
from dfply import *
import warnings
warnings.filterwarnings('ignore')


# ## Data Scrape based on Console

# #### DS_Data
# from the website we learnt there would be 66 pages in total

# In[124]:


# Data Url that we'll scrape. 
## notice there are several pages of the table so we will need to make a list to scrape table from this website
## 66 pages in total
ds_url_list = list()
ds_url_head = "https://www.vgchartz.com/games/games.php?page="
ds_url_tail = "&console=DS&order=Sales&ownership=Both&direction=DESC&showpublisher=1&showreleasedate=1&showlastupdate=1&showvgchartzscore=1&showcriticscore=1&showuserscore=1&showshipped=1"
page_list = list(range(1,67))
for i in page_list:
    ds_url = ds_url_head + str(i) + ds_url_tail
    ds_url_list.append(ds_url)
    




# In[125]:


df_list = list()
for link in ds_url_list:
    time.sleep(random.uniform(2,10))
    page = requests.get(link)
   
    table = pd.read_html(page.text)
    df = table[6]
    df_list.append(df)
   


# #### change the name of columns and concat these columns to a new dataframe
# 

# In[ ]:


new_df_list = list()
for dat in df_list:
    df = dat.set_axis(['Pos', 'Game', 'Game.1', 'Console', 'Publisher','VGChartz Score','Critic Score','User Score','Total Shipped','Release Date','Last Update','random'], axis=1, inplace=False)
    new_df_list.append(df)
   


# In[ ]:


ds_main_df = pd.concat(new_df_list,ignore_index=True)

ds_main_df.head()


# #### save the dataframe to csv file

# In[ ]:


ds_main_df.to_csv('ds_scrape.csv', index=False) 


# ## Now we repeat the same process for the other consoles

# Since the original website is very unstable, we break the pages to sections to scrape

# #### PS2

# In[78]:


# Data Url that we'll scrape. 
## notice there are several pages of the table so we will need to make a list to scrape table from this website
## 72 pages in total
ps2_url_list = list()
ps2_url_head = "https://www.vgchartz.com/games/games.php?page="
ps2_url_tail = "&console=PS2&order=Sales&ownership=Both&direction=DESC&showpublisher=1&showreleasedate=1&showlastupdate=1&showvgchartzscore=1&showcriticscore=1&showuserscore=1&showshipped=1"
page_list = list(range(1,73))
for i in page_list:
    ps2_url = ps2_url_head + str(i) + ps2_url_tail
    ps2_url_list.append(ps2_url)
    



# In[79]:


df_list = list()
for link in ps2_url_list[0:30]:
    time.sleep(random.uniform(3,10))
    page = requests.get(link)
    table = pd.read_html(page.text)
    df = table[6]
    df_list.append(df)
    


# In[81]:


df_list1 = list()
for link in ps2_url_list[31:50]:
    time.sleep(random.uniform(3,10))
    page = requests.get(link)
    table = pd.read_html(page.text)
    df = table[6]
    df_list1.append(df)
    


# In[82]:


df_list2 = list()
for link in ps2_url_list[51:71]:
    time.sleep(random.uniform(3,10))
    page = requests.get(link)
    table = pd.read_html(page.text)
    df = table[6]
    df_list2.append(df)
    


# In[83]:


df_list_ps2 = df_list + df_list1 + df_list2


# In[84]:


new_df_list = list()
for dat in df_list_ps2:
    df = dat.set_axis(['Pos', 'Game', 'Game.1', 'Console', 'Publisher','VGChartz Score','Critic Score','User Score','Total Shipped','Release Date','Last Update','random'], axis=1, inplace=False)
    new_df_list.append(df)


# In[85]:


ps2_main_df = pd.concat(new_df_list,ignore_index=True)

ps2_main_df.head()


# In[86]:


ps2_main_df.to_csv('ps2_scrape.csv', index=False) 


# #### PS3

# In[63]:


# Data Url that we'll scrape. 
## notice there are several pages of the table so we will need to make a list to scrape table from this website
## 38 pages in total
ps3_url_list = list()
ps3_url_head = "https://www.vgchartz.com/games/games.php?page="
ps3_url_tail = "&console=PS3&order=Sales&ownership=Both&direction=DESC&showpublisher=1&showreleasedate=1&showlastupdate=1&showvgchartzscore=1&showcriticscore=1&showuserscore=1&showshipped=1"
page_list = list(range(1,39))
for i in page_list:
    ps3_url = ps3_url_head + str(i) + ps3_url_tail
    ps3_url_list.append(ps3_url)
    



# In[70]:


ps3_url_list[37]


# In[72]:


df_list = list()
for link in ps3_url_list[0:20]:
    time.sleep(random.uniform(3,10))
    page = requests.get(link)
   
    table = pd.read_html(page.text)
    df = table[6]
    df_list.append(df)
    


# In[73]:


df_list1 = list()
for link in ps3_url_list[21:37]:
    time.sleep(random.uniform(3,10))
    page = requests.get(link)
    table = pd.read_html(page.text)
    df = table[6]
    df_list1.append(df)
    


# In[74]:


df_list_ps3 = df_list + df_list1


# In[75]:


new_df_list = list()
for dat in df_list_ps3:
    df = dat.set_axis(['Pos', 'Game', 'Game.1', 'Console', 'Publisher','VGChartz Score','Critic Score','User Score','Total Shipped','Release Date','Last Update','random'], axis=1, inplace=False)
    new_df_list.append(df)


# In[76]:


ps3_main_df = pd.concat(new_df_list,ignore_index=True)

ps3_main_df.head()


# In[77]:


ps3_main_df.to_csv('ps3_scrape.csv', index=False)


# #### Wii

# In[90]:


# Data Url that we'll scrape. 
## notice there are several pages of the table so we will need to make a list to scrape table from this website
## 34 pages in total
wii_url_list = list()
wii_url_head = "https://www.vgchartz.com/games/games.php?page="
wii_url_tail = "&console=wii&order=Sales&ownership=Both&direction=DESC&showpublisher=1&showreleasedate=1&showlastupdate=1&showvgchartzscore=1&showcriticscore=1&showuserscore=1&showshipped=1"
page_list = list(range(1,35))
for i in page_list:
    wii_url = wii_url_head + str(i) + wii_url_tail
    wii_url_list.append(wii_url)
    


# In[88]:


df_list1 = list()
for link in wii_url_list[0:20]:
    time.sleep(random.uniform(3,10))
    page = requests.get(link)
   
    table = pd.read_html(page.text)
    df = table[6]
    df_list1.append(df)
    


# In[91]:


df_list2 = list()
for link in wii_url_list[21:33]:
    time.sleep(random.uniform(3,10))
    page = requests.get(link)
   
    table = pd.read_html(page.text)
    df = table[6]
    df_list2.append(df)


# In[92]:


df_list_wii = df_list1 + df_list2


# In[93]:


new_df_list = list()
for dat in df_list_wii:
    df = dat.set_axis(['Pos', 'Game', 'Game.1', 'Console', 'Publisher','VGChartz Score','Critic Score','User Score','Total Shipped','Release Date','Last Update','random'], axis=1, inplace=False)
    new_df_list.append(df)


# In[94]:


wii_main_df = pd.concat(new_df_list,ignore_index=True)

wii_main_df.head()


# In[96]:


wii_main_df.to_csv('wii_scrape.csv', index=False)


# ## Now we scrape the data from FANDOM for the total sales through out the years

# In[132]:


link = "https://vgsales.fandom.com/wiki/Video_game_industry"
page = requests.get(link)
table = pd.read_html(page.text)


# In[133]:



df = table[1]
df.head()


# In[134]:


#change the column name for convenience
df = df.rename(columns = {"Inflation-adjusted revenue(2012 US dollars)[109]":"Revenue"})


# In[135]:


#use only the adjusted revenue
#use the lagged value for prediction
rev= df["Revenue"].str.split("$", n = 1, expand = True) 
df["Rev"] = rev[1]
rev = df["Rev"].str.split(" ", n = 1, expand = True) 
df["Rev"] = rev[0]
df["Year"] = df["Year"] +1
df = df.filter(items = ["Year","Rev"])
df.head()


# In[136]:


df.to_csv('money_sale.csv', index=False)


# In[ ]:




