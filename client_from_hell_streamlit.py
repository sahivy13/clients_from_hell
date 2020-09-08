import requests, time
from bs4 import BeautifulSoup as bs
import re
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
#from nltk.corpus import stopwords
import re
from collections import Counter
import functools
import operator
import string
from sklearn.model_selection import train_test_split
import re
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.palettes import Category20
from bokeh.transform import factor_cmap
from PIL import Image



st.title("Client From Hell Classifier")

st.text('The data used for this projects has was scraped from the following URL: https://clientsfromhell.net/')


# image = Image.open('bagw.jpg')
# st.image(image,width=560,use_column_width=False,format='PNG')

#Data scraper
def get_categories():
    url = 'https://clientsfromhell.net/'
    html = requests.get(url)
    response = bs(html.content)
    get_items = [category for category in response.find_all('li', {'class':'flex items-center'})]
    categories = ['Dunces','Criminals','Deadbeats','Racists','Homophobes','Sexist','Frenemies','Cryptic','Ingrates','Chaotic Good']
    category_pair = []
    for item in get_items:
        href = item.find('a').get('href')
        item_name = re.sub('\\n','',item.text)
        pair = (item_name, href)
        if item_name in categories:
            category_pair.append(pair)
    return list(set(category_pair))

list_catetories = get_categories()

#URL Category Creator
list_url_patters = []

def URL_CATEGORY_CREATOR(x):
    for cat in list_catetories:
        pattern = 'https://clientsfromhell.net'+cat[1]+'page/' # regex pattern for the urls to scrape
        list_url_patters.append((pattern,cat[0]))
    return list_url_patters
        
url_cat = URL_CATEGORY_CREATOR(list_catetories)



#Pages iterator
def PAGE_NUM_CREATOR(x):
    list_url_num =[]
    for url in url_cat:
        html = requests.get(url[0]+'1')
        response = bs(html.content)
        list_items = response.find_all('a',{'class':'page-numbers'})

        len_=len(list_items)-2
        max_pag=list_items[len_].text
        list_url_num.append((url[0],max_pag,url[1]))
    return list_url_num

list_all = PAGE_NUM_CREATOR(url_cat)

#IRONHACK SPIDER CLASS 
class IronhackSpider:
    
    def __init__(self, url_pattern, pages_to_scrape=1, sleep_interval=-1, content_parser=None):
        self.url_pattern = url_pattern
        self.pages_to_scrape = pages_to_scrape
        self.sleep_interval = sleep_interval
        self.content_parser = content_parser
  
    def scrape_url(self, url):
        response = requests.get(url)
        result = self.content_parser(response.content)
        return result
#         self.list_results(result)
        
#     def list_results(self, r):
#         list_pages = []
#         list_pages.append(r)
#         return list_pages

#     def output_results(self, r):
#         print(r)

#     def kickstart(self):
#         for i in range(1, self.pages_to_scrape+1):
#             self.scrape_url(self.url_pattern % i)
            
    def kickstart(self):
        list_pages = []
        for i in range(1, self.pages_to_scrape+1):
            list_pages.append(self.scrape_url(self.url_pattern % i))            
        return list_pages


def content_parser(content):
    return content

def case_parser(content):
    all_content = bs(content, "html.parser")
    pre_content = all_content.select('div [class="w-blog-post-content"] > p')
    
    case=[]
    
    for i, el in enumerate(pre_content):
        text = el.text
        case.append(text)

    return case

# Instantiate the IronhackSpider class
html_cont_dict = {}

for URL_PATTERN, PAGES_TO_SCRAPE, CAT in list_all:
#     print(URL_PATTERN, PAGES_TO_SCRAPE, CAT)
    my_spider = IronhackSpider(URL_PATTERN+'%s/', int(PAGES_TO_SCRAPE), content_parser=case_parser)
    # Start scraping jobs
    content = my_spider.kickstart()
    html_cont_dict.update({CAT: content})
    
#Saves scraper output to CSV
import csv
with open('client_from_hell_dic.csv', 'w') as f:
    for key in html_cont_dict.keys():
        f.write("%s,%s\n"%(key,html_cont_dict[key]))

#Stopwords 
stopwords = ["mustn't", 'have', 'now', "shouldn't", 'down', 'theirs', 
             'where', 'doing', 'who', 'is', 'how', 'which', 'up', 'if', 
             'nor', 'y', 'a', 'couldn', "weren't", 'your', 'in', 'themselves', 
             'only', 'both', 'will', 'own', 'we', 'by', 'these', 'into', 'once', 
             'her', 'when', 'o', 's', 've', 'again', 'mustn', 'they', 't', 'aren', 
             'his', "haven't", "won't", 'same', 'not', 'there', 'hadn', 'has', 
             'can', "didn't", 'those', 'then', 'this', 'shan', 'such', 'wasn', 
             'of', 'after', 'through', 'from', "you'd", 'shouldn', "isn't", 'being', 
             'too', "that'll", 'between', 'had', 'yours', 'very', 'i', 'because', 
             'him', 'are', "needn't", 'before', 'wouldn', 'my', 'what', 'the', 'during', 
             'isn', 'weren', 'against', 'don', 'off', 'll', 'am', 'our', 'should', 
             "should've", "you've", 'hers', 'their', 'more', 'than', 'himself', 'itself', 
             'with', 'just', "hasn't", 'he', 'any', 'to', "couldn't", "you'll", 'on', 'she', 
             "she's", 'ma', 'an', 'haven', 'was', 'few', 'each', 'that', 'further', 'didn', 
             "hadn't", "it's", 'over', 'you', 'me', 'about', 'mightn', 'most', 'be', 'so', 
             'won', 'them', 'why', 'needn', 'or', 'above', "doesn't", 'd', 'yourselves', 
             'ours', 'herself', 'as', 'some', 'under', "don't", 'at', 'all', "wouldn't", 'it', 
             'does', 'but', 'below', 'here', 'other', 'whom', 'ourselves', "you're", 'did', 'do', 
             'while', "aren't", 'having', 're', 'ain', 'were', 'hasn', 'no', 'out', 'and', 'for', 
             'yourself', 'been', 'm', 'myself', 'its', 'until', "shan't", "wasn't", "mightn't", 'doesn', 
             "you’re", "aren’t"]

#Data preprocessing

df_clients_og = pd.DataFrame.from_dict(html_cont_dict, orient = 'index').fillna('')
df_clients_og = df_clients_og.T

def stem(a):
    p = PorterStemmer()
    a = [p.stem(word) for word in a]
    return a

# df_clients_og = pd.DataFrame.from_dict(html_cont_dict, orient = 'index').fillna('')
# df_clients_og = df_clients_og.T

for col in df_clients_og:
#     print(col)
        
    for i,list_ in enumerate(df_clients_og[col]):
        sub_list=[]

        for item in list_:
            if item.startswith('Client:'):
                sub_list.append(item)

        df_clients_og[col][i] = sub_list
#         print(sub_list)
        
punc_list = [x for x in string.punctuation]

for col in df_clients_og:
#     print(col)
        
    for i,list_ in enumerate(df_clients_og[col]):
        sub_list = [x.replace('\xa0', ' ') for x in df_clients_og[col][i]]
        sub_list = [x.replace('\n', ' ') for x in sub_list]
        sub_list = [x.replace('Client: ', '') for x in sub_list]
        for punc in punc_list:
            sub_list = [x.replace(punc, '') for x in sub_list]
        sub_list = [x.replace('—', '') for x in sub_list]
        sub_list = [x.replace('   ', ' ') for x in sub_list]
        sub_list = [x.replace('  ', ' ') for x in sub_list]
        sub_list = [x.rstrip() for x in sub_list]
        
        df_clients_og[col][i] = sub_list
        
for col in df_clients_og:
#     print(col)
        
    for i,list_ in enumerate(df_clients_og[col]):
        sub_list = [x.split(' ') for x in list_]
        
        df_clients_og[col][i] = sub_list
        df_clients_og[col][i] = [word.lower() for words in df_clients_og[col][i] for word in words if len(word) != 1]
        df_clients_og[col][i] = [re.sub(r'^(.)\1+', r'\1', word)  for word in df_clients_og[col][i]]

        df_clients_og[col][i] = [word.replace("’", "'") for word in df_clients_og[col][i]]
#         df_clients_og[col][i] = [word.replace("ing", "") for word in df_clients_og[col][i]]
#         df_clients_og[col][i] = [word.replace("ed", "") for word in df_clients_og[col][i]]
        df_clients_og[col][i] = [word.rstrip("'") for word in df_clients_og[col][i]]
        
        
        
        df_clients_og[col][i] = [word for word in df_clients_og[col][i] if word not in stopwords]
        df_clients_og[col][i] = [word for word in df_clients_og[col][i] if word.isalpha() == True]
        df_clients_og[col][i] = [word for word in df_clients_og[col][i] if len(word) != 1]
        df_clients_og[col][i] = stem(df_clients_og[col][i])
        
        
        
        df_clients_og[col][i] = dict(Counter(df_clients_og[col][i]))

#Reshaping the Data into a dictionary to transform it into a DataFrame        
df_final = df_clients_og.T

df_final.columns = [str(col) for col in df_final.columns]

df_final.reset_index(inplace = True)
df_final.rename(columns = {'index':'category'}, inplace = True)
# df_final.head()

df_cases = pd.DataFrame(columns = ['category', 'case'])

# df_final[['category', '0']].rename(columns = {'0':'case'})

for col in df_final:
    if col != 'category':
        df_cases = df_cases.append(df_final[['category', col]].rename(columns = {col:'case'}))

df_cases.reset_index(drop = True, inplace = True)

for i, row in enumerate(df_cases['case']):
    if row == {}:
        df_cases.drop(index = i, inplace = True)
        
df_cases.reset_index(drop = True, inplace = True)

df_count = pd.DataFrame()

for i in range(df_cases['case'].shape[0]):
#     print(i)
    df_count = df_count.append(pd.DataFrame(df_cases['case'][i], index=[0]))
    
df_count.reset_index(drop = True, inplace = True)

df_count['category'] = df_cases['category']

df_count.to_csv('final_dataframe.csv')
df_final = pd.read_csv('final_dataframe.csv')
df_final.drop(columns = 'Unnamed: 0', inplace = True)
df_new_final = df_final.fillna('0')

features = df_new_final[[col for col in df_new_final.columns if col != 'category']]
features = features.astype(int)

target = df_new_final[['category']]
target

for i in range(target.shape[0]):
    target['category'] = [x.replace(x, '1') if x == 'Deadbeats' else x.replace(x,'0') for x in target['category']]
    
target = target.astype(int)

#Training Model
def model_score_mul(x):
    import time
    
    score = x.score(X_test, y_test_mul)*100
    if score >= 50.0:
        print('The score for this algorithm is (drum roll please!!!): ')
        time.sleep(3)
        print('.... processing')
        time.sleep(3)
        print('....')
        time.sleep(3)
        print('*looks at you confused*')
        time.sleep(3)
        print('oh yeah, you are waiting for the score... LOL')
        print('score: ',score,'%')
        print("DON'T GET COCKY NOW!!! KEEP MAKING IT BETTER!")
    elif score < 50.0:
        print('The score for this algorithm is: ',score,'%')
        print('Tough luck buddy... try something else "/')
        
#Train data split        
target_mul = df_new_final[['category']]
target_mul

X_train, X_test, y_train_mul, y_test_mul = train_test_split(
    features, # Features (X)
    target_mul, # Target (y)
    test_size = .2,
    random_state = 42
)
#Training Model
st.text('Logistic Regression Classification Algorithm Score')
from sklearn.linear_model import LogisticRegression

log_regr = LogisticRegression(solver = 'lbfgs')
log_regr.fit(X_train, y_train_mul.values.ravel())
print(model_score_mul(log_regr))

#
st.text('MultinomialNB Classification Algorithm Score')
from sklearn.naive_bayes import MultinomialNB

multi = MultinomialNB()
multi.fit(X_train, y_train_mul.values.ravel())
print(model_score_mul(multi))
