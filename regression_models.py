import pandas as pd
import numpy as np
import os
import copy

import streamlit as st

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

# Model saving
import pickle

# Math function needed for models
from math import sqrt

# Metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Model preprocessing
# from sklearn.feature_extraction.text import TfidfVectorizer -> imported by another .py file
from sklearn.preprocessing import MinMaxScaler

# # Stating random seed
# np.random.seed(42)

def all_num_models_fitting(X_train, y_train):

    if os.path.isfile('rfc'):

        list_ = ['log_regr', 'knn', 'multi', 'rfc']

        with open (list_[0], 'rb') as f :
            log_regr = pickle.load(f)

        with open (list_[1], 'rb') as f :
            knn = pickle.load(f)

        with open (list_[2], 'rb') as f :
            multi = pickle.load(f)

        with open (list_[3], 'rb') as f :
            rfc = pickle.load(f)

    else:
        log_regr = LogisticRegression(solver = 'lbfgs')
        log_regr.fit(X_train, y_train.values.ravel())
        # d_log_regr = copy.deepcopy(log_regr)

        knn = KNeighborsClassifier(n_neighbors = 3) # k = 5 by default
        knn.fit(X_train, y_train.values.ravel())
        # d_knn = copy.deepcopy(knn)

        multi = MultinomialNB()
        multi.fit(X_train, y_train.values.ravel())
        # d_multi = copy.deepcopy(multi)
        
        rfc = RandomForestClassifier(max_depth=10, random_state=42)
        rfc.fit(X_train, y_train.values.ravel())
        # d_rfc = copy.deepcopy(rfc)

        list_ = [('log_regr', log_regr), ('knn', knn), ('multi', multi), ('rfc', rfc)]

        for mod in list_:
            with open (mod[0], 'wb') as f:
                pickle.dump(mod[1],f) 

            # with open (mod[0], 'rb') as f :
            #     mod[1] = pickle.load(f)
    
    return log_regr, knn, multi, rfc

def all_bool_models_fitting(X_train, y_train):

    if os.path.isfile('guassian'):

        list_ = ['bernoulli', 'guassian']

        with open (list_[0], 'rb') as f :
            bernoulli = pickle.load(f)

        with open (list_[1], 'rb') as f :
            guassian = pickle.load(f)

    else: 

        bernoulli = BernoulliNB().fit(X_train, y_train.values.ravel())
        # d_bernoulli = copy.deepcopy(bernoulli)
        
        guassian = GaussianNB().fit(X_train, y_train.values.ravel())
        # d_guassian = copy.deepcopy(guassian)
        
        list_ = [('bernoulli', bernoulli), ('guassian', guassian)]

        for mod in list_:
            with open (mod[0], 'wb') as f:
                pickle.dump(mod[1],f)
            
            # with open (mod[0], 'rb') as f :
            #     mod[1] = pickle.load(f)
    
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

def new_run_all_models_and_score_k_fold(df):
    
    model_names = ['**Log Regression**', '**KNN**', '**Multinomial**', '**Random Forest**', '**Bernoulli**', '**Gaussian**']
    
    metrics_names = [
        'R2: ', 'MSE: ', 'RMSE: ', 'MAE: ',
        'Accuracy: ', 'Balanced Acc: ', 'Precision: ',
        'Recall: ', 'F1 Score: '
    ]

    model_dict = {}



    for i, model in enumerate(model_names):
        if i < 4:

            # model_dict[model_names[i]] = 

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