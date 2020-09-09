import warnings
warnings.filterwarnings("ignore")

import functools
import streamlit as st

st.title('Clients From Hell project')

import requests, time
from bs4 import BeautifulSoup as bs
import re
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import string

# import operator

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from math import sqrt

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import plotly.figure_factory as ff
import plotly.express as px

# import warnings
# warnings.filterwarnings('ignore')

@st.cache(suppress_st_warning=True)

# url = "https://clientsfromhell.net/"

#---Scrapper---

def pipe(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

def pipe_1(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

def pipe_2(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

def pipe_3(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

def get_categories(url):
    html = requests.get(url)
    response = bs(html.content, features="html.parser")
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
        response = bs(html.content, "html.parser")
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


# def content_parser(content):
#     return content


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

#---Preprocessing---

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
    df_cases.reset_index(drop = True, inplace = True) #ADDED

    return df_cases

def df_creator(dic_):
    df_ = pd.DataFrame.from_dict(dic_, orient = 'index').fillna('').transpose()
    return df_

def catetory_replacer(df, col = 'category'):

    # dic_cat = {}

    # for i, cat in enumerate(list(df[col].unique())):
    #     dic_cat[cat] = i

    dic_cat = {
        "Deadbeats" : 1,
        'Dunces' : 0,
        'Criminals' : 0,
        'Racists' : 0,
        'Homophobes' : 0,
        'Sexist' : 0,
        'Frenemies' : 0,
        'Cryptic' : 0,
        'Ingrates' : 0,
        'Chaotic Good' : 0
    } 

    df[col].replace(to_replace = dic_cat, inplace = True)
    
    return df, dic_cat

#---Regression tools

def t_t_split(df, target_col = 'category'):
    
    features = df[[col for col in df.columns if col != target_col]]
    target = df[[target_col]]

    
    X_train, X_test, y_train, y_test = train_test_split(
        features, # Features (X)
        target, # Target (y)
        test_size = .2,
        random_state = 42
    )
    return X_train, X_test, y_train, y_test

def list_split(string_):
    list_ = string_.split()
    return list_

def convert_to_word_col(df, case_col = 'case', target_col = 'category'):
    
    series_ = df[case_col].apply(lambda x: list_split(x))
    series_ = series_.apply(lambda x: dict(Counter(x)))
    
    df_series = pd.DataFrame(series_)
    df_count = pd.DataFrame()
    
    for i in range(df_series.shape[0]):
        df_count = df_count.append(pd.DataFrame(df_series[case_col][i], index=[0]))
        
    df_count.reset_index(drop = True, inplace = True)
        
    df_ = df[[target_col]]
    df_ = df_.merge(df_count, left_index=True, right_index= True)
    df_.fillna(0, inplace = True)
    
    return df_

def all_num_models_fitting(X_train, y_train):

    log_regr = LogisticRegression(solver = 'lbfgs')
    log_regr.fit(X_train, y_train.values.ravel())

    knn = KNeighborsClassifier(n_neighbors = 3) # k = 5 by default
    knn.fit(X_train, y_train.values.ravel())

    multi = MultinomialNB()
    multi.fit(X_train, y_train.values.ravel())

    rfc = RandomForestClassifier(max_depth=10, random_state=42)
    rfc.fit(X_train, y_train.values.ravel())
    
    return log_regr, knn, multi, rfc

def all_bool_models_fitting(X_train, y_train):

    bernoulli = BernoulliNB().fit(X_train, y_train.values.ravel())

    guassian = GaussianNB().fit(X_train, y_train.values.ravel())
    
    return bernoulli, guassian

def model_score(model, X_test, y_test):
    
    score = model.score(X_test, y_test)*100
    
    if score >= 50.0:
        
        print('Score: ',score,'%')
        print("DON'T GET COCKY NOW!!! KEEP MAKING IT BETTER!")
        print('')
    elif score < 50.0:
        
        print('Score: ',score,'%')
        print("Your algorithm stinks so much, I could toss a coin and make better predictions =P...")
        print('')
    
    return score

def predict(model, X_test):
    prediction = model.predict(X_test)
    return prediction

def convert_to_tfidf(df, case_col = 'case', target_col = 'category'):
    
    tfidf = TfidfVectorizer()
    word_count_vectors = tfidf.fit_transform(df[case_col].values).todense().tolist()
    
    features = pd.DataFrame(
    data = word_count_vectors,
    columns = tfidf.get_feature_names()
    )

    df_ = features.merge(df[target_col], left_index=True, right_index= True)
    
    return df_

def under_sampling_to_2_col_by_index(df, high = 0, low = 1, col_name = 'category'):
    
    # def convert_3(int_):
    #     if int_ > 4:
    #         return 2
    #     elif int_ == 4:
    #         return 1
    #     elif int_ == 0:
    #         return 0
    #     elif int_ == 1: 
    #         return 2
    #     elif int_ == 2: 
    #         return 2
    #     elif int_ == 3: 
    #         return 0
    
    # df = df_[[col for col in df_.columns if col != col_name]]
    
    # df[col_name] = df_[col_name].apply(lambda x: convert_3(x))
    
    low_size = len(df[df[col_name] == low])
    
    high_indices = df[df[col_name] == high].index
    
    # mid_indices = df[df[col_name] == mid].index
    
    low_indices = df[df[col_name] == low].index
    
    random_high_indices = np.random.choice(high_indices, low_size, replace=False)
    
    # random_mid_indices = np.random.choice(mid_indices, low_size, replace=False)
    
    under_sample_indices = np.concatenate([random_high_indices,low_indices]) #,random_mid_indices
    
    under_sample = df.loc[under_sample_indices]
    
    under_sample = under_sample.reset_index(drop = True)
    
    return under_sample

def evaluate_model(model, train_X, test_X, train_y, test_y):
    
    model = model.fit(train_X, train_y)
    score = model.score(test_X, test_y)
    #     print(f"Accuracy: {round(score, 2)}")
    return model, score

def k_fold_score_new(df, model_name, target = 'category'):
    scores = []
    r2_scores = []
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    acc_scores = []
    bacc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []

    features = df[[col for col in df if col != target]]
    target = df[target]
    
    n= 10
    kf = KFold(n_splits = n, random_state = 42)
    for train_i, test_i in kf.split(df):
        
        X_train = features.iloc[train_i]
        X_test = features.iloc[test_i]
        y_train = target.iloc[train_i]
        y_test = target.iloc[test_i]
        
        X_train_bool = X_train.astype('bool')
        X_test_bool = X_test.astype('bool')
        
        models = all_num_models_fitting(X_train, y_train) #log_regr, knn, multi, rfc
        models = models + all_bool_models_fitting(X_train_bool, y_train) #adding ber, gau
        
        model_names = {
            '**Log Regression**': 0, '**KNN**': 1,
            '**Multinomial**': 2, '**Random Forest**': 3,
            '**Bernoulli**': 4, '**Gaussian**': 5
        }

        # model_names.get(model_name)
        
        if model_names.get(model_name) < 4:

            score = models[model_names.get(model_name)].score(X_test, y_test) #returns score 
            scores.append(score)

            prediction = predict(models[model_names.get(model_name)], X_test)

            r2 = r2_score(y_test, prediction)
            r2_scores.append(r2)

            mse = mean_squared_error(y_test, prediction)
            mse_scores.append(mse)

            rmse = sqrt(mse)
            rmse_scores.append(rmse)

            mae = mean_absolute_error(y_test, prediction)
            mae_scores.append(mae)

            acc = accuracy_score(y_test, prediction)
            acc_scores.append(acc)

            bacc = balanced_accuracy_score(y_test, prediction)
            bacc_scores.append(bacc)

            prec = precision_score(
                y_test,
                prediction,
                pos_label = 2,
                average = 'weighted'
            )
            prec_scores.append(prec)
            
            rec = recall_score(
            y_test,
            prediction,
            pos_label = 2,
            average = 'weighted'
            )
            rec_scores.append(rec)

            f1 = f1_score(
                y_test,
                prediction,
                pos_label = 2,
                average = 'weighted'
            )
            f1_scores.append(f1)
            
            # return r2, mse, rmse, mae, acc, bacc, prec, rec, f1
        else:
            score = models[model_names.get(model_name)].score(X_test_bool, y_test) #returns score 
            scores.append(score)

            prediction = predict(models[model_names.get(model_name)], X_test_bool)

            r2 = r2_score(y_test, prediction)
            r2_scores.append(r2)

            mse = mean_squared_error(y_test, prediction)
            mse_scores.append(mse)

            rmse = sqrt(mse)
            rmse_scores.append(rmse)

            mae = mean_absolute_error(y_test, prediction)
            mae_scores.append(mae)

            acc = accuracy_score(y_test, prediction)
            acc_scores.append(acc)

            bacc = balanced_accuracy_score(y_test, prediction)
            bacc_scores.append(bacc)

            prec = precision_score(
                y_test,
                prediction,
                pos_label = 2,
                average = 'weighted'
            )
            prec_scores.append(prec)
            
            rec = recall_score(
            y_test,
            prediction,
            pos_label = 2,
            average = 'weighted'
            )
            rec_scores.append(rec)

            f1 = f1_score(
                y_test,
                prediction,
                pos_label = 2,
                average = 'weighted'
            )
            f1_scores.append(f1)

    def avg_score(x):
        avg = sum(x)/len(x)
        return avg

    score_avg = avg_score(scores)
    r2_avg = avg_score(r2_scores)
    mse_avg= avg_score(mse_scores)
    rmse_avg = avg_score(rmse_scores)
    mae_avg = avg_score(mae_scores)
    acc_avg = avg_score(acc_scores)
    bacc_avg = avg_score(bacc_scores)
    prec_avg = avg_score(prec_scores)
    rec_avg = avg_score(rec_scores)
    f1_avg = avg_score(f1_scores)


    return (
        score_avg, r2_avg, mse_avg,
        rmse_avg, mae_avg, acc_avg,
        bacc_avg, prec_avg, rec_avg, f1_avg
        )

def rescale_numbers(df, scaler = MinMaxScaler):
    for col in df:
        if col != 'category':
            if df[col].dtype in ['int64', 'float64']:
                numbers = df[col].astype(float).values.reshape(-1, 1)
                df[col] = scaler().fit_transform(numbers)
            
    return df

def run_all_models_and_score_k_fold(df):
    
    model_names = ['**Log Regression**', '**KNN**', '**Multinomial**', '**Random Forest**', '**Bernoulli**', '**Gaussian**']
    
    metrics_names = [
        'R2: ', 'MSE: ', 'RMSE: ', 'MAE: ',
        'Accuracy: ', 'Balanced Acc: ', 'Precision: ',
        'Recall: ', 'F1 Score: '
    ]

    for i, model in enumerate(model_names):
        if i < 4:

            st.write(model_names[i])
            st.write('')
            st.write('kfold score: ',k_fold_score_new(df, model_names[i])[0]*100,'%') #prints score kfold
            st.write('')

            for j, name in enumerate(metrics_names):
                  st.write(name, k_fold_score_new(df, model_names[i])[j-1]) #r2, mse, rmse, mae, acc, bacc, prec, rec, f1

            st.write('')

        else:

            st.write(model_names[i])
            st.write('')
            st.write('kfold score: ',k_fold_score_new(df, model_names[i])[0]*100,'%') #prints score kfold
            st.write('')
            
            for j, name in enumerate(metrics_names):
                  st.write(name, k_fold_score_new(df, model_names[i])[j-1]) #r2, mse, rmse, mae, acc, bacc, prec, rec, f1
            st.write('')


#NEW ADDED TO CREATE HISTOGRAM
def hist_of_target_creator(df, target = 'category'):
 
    fig = px.histogram(df, x = target, y = target) #, color="sex", marginal="rug", hover_data=tips.columns)

    # Plot!
    st.plotly_chart(fig)

    return df

def streamlit_pipe_write_before(df_):
    st.write(f"**Before Undersampling (This shows 1 for Deadbeats, and 0 for non-Deadbeats)**")
    return(df_)
    
def streamlit_pipe_write_after(df_):
    st.write(f"**After**")
    return(df_)

def streamlit_pipe_write_paragraph(url):
    st.write(f"This project involves supervised machine learning algorithms classifier,")
    st.write(f"where we pull data from the website 'https://clientsfromhell.net/'")
    st.write(f"Once we have the data, we go case by case in each category and only pull")
    st.write(f"any dialogues, specifically what the client would say per case.")
    st.write(f"the rest of the process can be explained by the line of code above.")
    return(url)

#      ****FINAL PIPELINE****

with st.echo():
    pipe_1( #Overall Pipe
        pipe_2( #Scrapper and Pipe creating dataframe
        "https://clientsfromhell.net/", #URL
        streamlit_pipe_write_paragraph, #Writting in streamlit
        get_categories, url_categroy_creator, page_num_creator, initialize_scraping, #Gets dictionary from website
        df_creator, #Creating dataframe out of dictionary
        cleaning, #Cleaning dataframe
        hist_of_target_creator, #Creates a Histogram of the data distribution for the categories
        catetory_replacer #Replacing category string values with integers
    )[0],#Data Frame: 0, Dictionary of Categories: 1
    streamlit_pipe_write_before, #Strimlit writing
    hist_of_target_creator, #Creates histogram of Deadbeats vs non-Deadbeats
    under_sampling_to_2_col_by_index, #Undersampling in order to have a better distribution
    streamlit_pipe_write_after, #Streamlit writing
    hist_of_target_creator, #Creates histogram of Deadbeats vs non-Deadbeats after undersampling
    convert_to_tfidf, # Converting sentences to word columns with tfidf method
    run_all_models_and_score_k_fold #Running and Scoring all models with kfold score included
)

# pipe_1( #Overall Pipe
#     pipe_2(
#         pipe("https://clientsfromhell.net/", #URL
#         get_categories, url_categroy_creator, page_num_creator, initialize_scraping), #Gets dictionary from website
#         df_creator, #Creating dataframe out of dictionary
#         cleaning, #Cleaning dataframe
#         hist_of_target_creator, #Creates a Histogram of the data distribution for the categories
#         catetory_replacer #Replacing category string values with integers
#     )[0],#Data Frame: 0, Dictionary of Categories: 1
#     streamlit_pipe_write_before, #Strimlit writing
#     hist_of_target_creator, #Creates histogram of Deadbeats vs non-Deadbeats
#     under_sampling_to_2_col_by_index, #Undersampling in order to have a better distribution
#     streamlit_pipe_write_after, #Streamlit writing
#     hist_of_target_creator, #Creates histogram of Deadbeats vs non-Deadbeats after undersampling
#     convert_to_tfidf, # Converting sentences to word columns with tfidf method
#     run_all_models_and_score_k_fold #Running and Scoring all models with kfold score included
# )


#      ****TESTING WITH SAVED VAR ****


# pipe_1( #Overall Pipe
#     pipe_2(var, #Dictionary from website
#         df_creator, #Creating dataframe out of dictionary
#         cleaning, #Cleaning dataframe
#         hist_of_target_creator,
#         catetory_replacer #Replacing category string values with integers
#     )[0],#Data Frame: 0, Dictionary of Categories: 1
#     streamlit_pipe_write_before,
#     hist_of_target_creator,
#     under_sampling_to_2_col_by_index, #Undersampling in order to have a better distribution
#     streamlit_pipe_write_after,
#     hist_of_target_creator,
#     convert_to_tfidf, # Converting sentences to word columns with tfidf method
#     run_all_models_and_score_k_fold #Running and Scoring all models with kfold score included
# )

# var = pipe("https://clientsfromhell.net/", get_categories, url_categroy_creator, page_num_creator, initialize_scraping)
# df_clients_og = pd.DataFrame.from_dict(var, orient = 'index').fillna('').transpose()
# df_test = cleaning(df_clients_og)



#      *** IN MEMORY OF THOSE FALLEN ***
#      AKA FORMULAS THAT DIDN'T MAKE IT BUT
#          WOULD LIKE TO REMEMBER :(



# def run_all_models_and_score(df):

#     X_train, X_test, y_train, y_test = t_t_split(df)
    
#     X_train_bool = X_train.astype(bool)
    
#     X_test_bool = X_test.astype(bool)

#     models = all_num_models_fitting(X_train, y_train) #log_regr, knn, multi, rfc
    
#     models = models + all_bool_models_fitting(X_train_bool, y_train) #adding ber, gau
    
#     metrics_names = ['R2: ', 'MSE: ', 'RMSE: ', 'MAE: ',
#                      'Accuracy: ', 'Balanced Acc: ', 'Precision: ',
#                      'Recall: ', 'F1 Score: ']
    
#     model_names = ['Log Regression', 'KNN', 'Multinomial', 'Random Forest', 'Bernoulli', 'Gaussian']
    
#     for i, model in enumerate(models):
#         if i < 4:
            
#             print(model_names[i])
#             print('')
#             model_score(model, X_test, y_test) #returns score and prints it
#             for j, name in enumerate(metrics_names):
#                   print(name, model_metrics(y_test, predict(model, X_test))[j]) #r2, mse, rmse, mae, acc, bacc, prec, rec, f1
#             print('')
#         else:
            
#             print(model_names[i])
#             print('')
#             model_score(model, X_test_bool, y_test) #returns score and prints it
#             for j, name in enumerate(metrics_names):
#                   print(name, model_metrics(y_test, predict(model, X_test_bool))[j]) #r2, mse, rmse, mae, acc, bacc, prec, rec, f1
#             print('')

# def k_fold_score(df, model_name, target = 'category'):
#     scores = []
#     features = df[[col for col in df if col != target]]
#     target = df[target]
    
#     n= 10
#     kf = KFold(n_splits = n, random_state = 42)
#     for train_i, test_i in kf.split(df):
        
#         X_train = features.iloc[train_i]
#         X_test = features.iloc[test_i]
#         y_train = target.iloc[train_i]
#         y_test = target.iloc[test_i]
        
#         X_train_bool = X_train.astype('bool')
#         X_test_bool = X_test.astype('bool')
        
#         models = all_num_models_fitting(X_train, y_train) #log_regr, knn, multi, rfc
#         models = models + all_bool_models_fitting(X_train_bool, y_train) #adding ber, gau
        
#         model_names = {
#             '**Log Regression**': 0, '**KNN**': 1,
#             '**Multinomial**': 2, '**Random Forest**': 3,
#             '**Bernoulli**': 4, '**Gaussian**': 5
#         }

#         metrics_names = ['R2: ', 'MSE: ', 'RMSE: ', 'MAE: ',
#                      'Accuracy: ', 'Balanced Acc: ', 'Precision: ',
#                      'Recall: ', 'F1 Score: ']

#         # model_names.get(model_name)
        
#         if model_names.get(model_name) < 4:
#             score = models[model_names.get(model_name)].score(X_test, y_test) #returns score 
#             scores.append(score)
#         else:
#             score = models[model_names.get(model_name)].score(X_test_bool, y_test) #returns score 
#             scores.append(score)
#     # print('number of folds for kfold :',n)
#     # st.write('number of folds for kfold :',n)
        
#     return sum(scores) / len(scores)

# def model_metrics(y_test, prediction):

    # r2 = r2_score(y_test, prediction)

    # mse = mean_squared_error(y_test, prediction)

    # rmse = sqrt(mean_squared_error(y_test, prediction))

    # mae = mean_absolute_error(y_test, prediction)

    # acc = accuracy_score(y_test, prediction)
    
    # bacc = balanced_accuracy_score(y_test, prediction)

    # prec = precision_score(
    #     y_test,
    #     prediction,
    #     pos_label = 2,
    #     average = 'weighted'
    # )
    
    # rec = recall_score(
    # y_test,
    # prediction,
    # pos_label = 2,
    # average = 'weighted'
    # )

    # f1 = f1_score(
    #     y_test,
    #     prediction,
    #     pos_label = 2,
    #     average = 'weighted'
    # )
    
    # return r2, mse, rmse, mae, acc, bacc, prec, rec, f1

#This function looked good but it sucks!!!
# def under_sampling_3_val_by_index(df_, high = 0, mid = 2, low = 1, col_name = 'category'):
    
#     def convert_3(int_):
#         if int_ > 4:
#             return 2
#         elif int_ == 4:
#             return 1
#         elif int_ == 0:
#             return 0
#         elif int_ == 1: 
#             return 2
#         elif int_ == 2: 
#             return 2
#         elif int_ == 3: 
#             return 0
    
#     df = df_[[col for col in df_.columns if col != col_name]]
    
#     df[col_name] = df_[col_name].apply(lambda x: convert_3(x))
    
#     low_size = len(df[df[col_name] == low])
    
#     high_indices = df[df[col_name] == high].index
    
#     mid_indices = df[df[col_name] == mid].index
    
#     low_indices = df[df[col_name] == low].index
    
#     random_high_indices = np.random.choice(high_indices, low_size, replace=False)
    
#     random_mid_indices = np.random.choice(mid_indices, low_size, replace=False)
    
#     under_sample_indices = np.concatenate([random_high_indices,random_mid_indices,low_indices])
    
#     under_sample = df.loc[under_sample_indices]
    
#     under_sample = under_sample.reset_index(drop = True)
    
#     return under_sample