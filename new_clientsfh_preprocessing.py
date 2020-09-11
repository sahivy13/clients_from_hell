import pandas as pd
import numpy as np
import regex as re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def df_creator(dic_):
    df_ = pd.DataFrame.from_dict(dic_, orient = 'index').fillna('').transpose()
    return df_

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

def catetory_replacer(df, col = 'category', mult = False, main_cat = "Deadbeats"):

    if mult == True: #--- MULTILABEL ---
        dic_cat = {}
        for i, cat in enumerate(list(df[col].unique())):
            dic_cat[cat] = i

    else: #--- CATEGORY VS. NOT CATEGORY ---
        dic_cat = {
            "Deadbeats" : 0,
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
        
        dic_cat[main_cat]  =  1

    df[col].replace(to_replace = dic_cat, inplace = True)
    
    return df, dic_cat

def under_sampling_to_2_col_by_index(df, high = 0, low = 1, col_name = 'category'):
    
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

def convert_to_tfidf(df, case_col = 'case', target_col = 'category'):
    
    tfidf = TfidfVectorizer()
    word_count_vectors = tfidf.fit_transform(df[case_col].values).todense().tolist()
    
    features = pd.DataFrame(
    data = word_count_vectors,
    columns = tfidf.get_feature_names()
    )

    df_ = features.merge(df[target_col], left_index=True, right_index= True)
    
    return df_

def data_to_csv(obj_df_dic):
    obj_df_dic[0].to_csv("scrapped_data.csv")
    df = obj_df_dic[0]
    return df

def data_from_csv(path):
    df = pd.read_csv(path)
    return df