import requests, time
from bs4 import BeautifulSoup as bs
import re
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import functools
import operator
import string
from sklearn.model_selection import train_test_split

def pipe(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

url = 'https://clientsfromhell.net/'

def get_categories(url):
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

def url_categroy_creator(list_categories):

    list_url_patters = []

    for cat in list_categories:
        pattern = 'https://clientsfromhell.net'+cat[1]+'page/' # regex pattern for the urls to scrape
        list_url_patters.append((pattern,cat[0]))

    return list_url_patters
        
def page_num_creator(url_category_list : list):
    list_url_num =[]
    for url in url_category_list:
        html = requests.get(url[0]+'1')
        response = bs(html.content)
        list_items = response.find_all('a',{'class':'page-numbers'})

        len_=len(list_items)-2
        max_pag=list_items[len_].text
        list_url_num.append((url[0],max_pag,url[1]))
    return list_url_num

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

def initialize_scraping(url_pagenum_cat_list : list):
    
    html_cont_dict = {}

    for URL_PATTERN, PAGES_TO_SCRAPE, CAT in url_pagenum_cat_list:

        my_spider = IronhackSpider(URL_PATTERN+'%s/', int(PAGES_TO_SCRAPE), content_parser=case_parser)

        content = my_spider.kickstart()
        
        html_cont_dict.update({CAT: content})
        
    return html_cont_dict

def stem(sentence : string):
    p = PorterStemmer()
    sentence = [p.stem(word) for word in sentence]
    return sentence

def cleaning(df : pd.DataFrame):
    
    for col in df:

        for i,list_ in enumerate(df[col]):
            
            sub_list=[]

            for item in list_:
                if item.startswith('Client:'):
                    sub_list.append(item)

            df[col][i] = sub_list
    

    punc_list = [x for x in string.punctuation]

    for col in df:

        for i,list_ in enumerate(df[col]):

            sub_list = [x.replace('\xa0|\n|Client: ', ' ') for x in df[col][i]]
            
            for punc in punc_list:
                sub_list = [x.replace(punc, '') for x in sub_list]
                
            sub_list = [x.replace('—|   |  ', '').rstrip() for x in sub_list]

            df[col][i] = sub_list
            

    for col in df:

        for i,list_ in enumerate(df[col]):

            sub_list = [x.split(' ') for x in list_]

            df[col][i] = sub_list
            df[col][i] = [word.lower() for words in df[col][i] for word in words if len(word) != 1]
            df[col][i] = [re.sub(r'^(.)\1+', r'\1', word)  for word in df[col][i]]
            df[col][i] = [word.replace("’", "'") for word in df[col][i]]
            df[col][i] = [word.replace("client", "") for word in df[col][i]]
            df[col][i] = [word.rstrip("'") for word in df[col][i]]

            df[col][i] = [word for word in df[col][i] if word not in stopwords.words('english')]
            df[col][i] = [word for word in df[col][i] if word.isalpha() == True]
            df[col][i] = [word for word in df[col][i] if len(word) != 1]
            df[col][i] = stem(df[col][i])

    
    df_final = df.transpose()

    df_final.columns = [str(col) for col in df_final.columns]

    df_final.reset_index(inplace = True)
    df_final.rename(columns = {'index':'category'}, inplace = True)

    df_cases = pd.DataFrame(columns = ['category', 'case'])

    for col in df_final:
        if col != 'category':
            df_cases = df_cases.append(df_final[['category', col]].rename(columns = {col:'case'}))

    df_cases.reset_index(drop = True, inplace = True)

    for i, row in enumerate(df_cases['case']):
        if row == []:
            df_cases.drop(index = i, inplace = True)

    df_cases['case'] = df_cases['case'].apply(lambda x: ' '.join(x))


    return df_cases

var = pipe(url, get_categories, url_categroy_creator, page_num_creator, initialize_scraping)
df_clients_og = pd.DataFrame.from_dict(var, orient = 'index').fillna('').transpose()
cleaning(df_clients_og)