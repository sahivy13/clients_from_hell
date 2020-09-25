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

# Cross Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


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

# Streamlit
import streamlit as st

# # Stating random seed
# np.random.seed(42)

def all_num_models_fitting(X_train, y_train):

    if os.path.isfile('rfc'):

        list_ = ['log_regr.pickle', 'knn.pickle', 'multi.pickle', 'rfc.pickle']

        with open (list_[0], 'rb') as f :
            log_regr = pickle.load(f)

        with open (list_[1], 'rb') as f :
            knn = pickle.load(f)

        with open (list_[2], 'rb') as f :
            multi = pickle.load(f)

        with open (list_[3], 'rb') as f :
            rfc = pickle.load(f)

        log_regr.fit(X_train, y_train.values.ravel())
        knn.fit(X_train, y_train.values.ravel())
        multi.fit(X_train, y_train.values.ravel())
        rfc.fit(X_train, y_train.values.ravel())

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

        list_ = [('log_regr.pickle', log_regr), ('knn.pickle', knn), ('multi.pickle', multi), ('rfc.pickle', rfc)]

        for mod in list_:
            with open (mod[0], 'wb') as f:
                pickle.dump(mod[1],f) 

            # with open (mod[0], 'rb') as f :
            #     mod[1] = pickle.load(f)
    
    return log_regr, knn, multi, rfc

def all_bool_models_fitting(X_train, y_train):

    if os.path.isfile('guassian.pickle'):

        list_ = ['bernoulli.pickle', 'guassian.pickle']

        with open (list_[0], 'rb') as f :
            bernoulli = pickle.load(f)

        with open (list_[1], 'rb') as f :
            guassian = pickle.load(f)

        bernoulli.fit(X_train, y_train.values.ravel())
        guassian.fit(X_train, y_train.values.ravel())

    else: 

        bernoulli = BernoulliNB().fit(X_train, y_train.values.ravel())
        # d_bernoulli = copy.deepcopy(bernoulli)
        
        guassian = GaussianNB().fit(X_train, y_train.values.ravel())
        # d_guassian = copy.deepcopy(guassian)
        
        list_ = [('bernoulli.pickle', bernoulli), ('guassian.pickle', guassian)]

        for mod in list_:
            with open (mod[0], 'wb') as f:
                pickle.dump(mod[1],f)
            
            # with open (mod[0], 'rb') as f :
            #     mod[1] = pickle.load(f)
    
    return bernoulli, guassian

def evaluate_model(model, train_X, test_X, train_y, test_y):
    
    model = model.fit(train_X, train_y)
    score = model.score(test_X, test_y)
    #     print(f"Accuracy: {round(score, 2)}")
    return model, score

def predict(model, X_test):
    prediction = model.predict(X_test)
    return prediction

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

            model = models[model_names.get(model_name)]
            score = model.score(X_test, y_test) #returns score 
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
                pos_label = 10,
                average = 'weighted'
            )
            prec_scores.append(prec)
            
            rec = recall_score(
            y_test,
            prediction,
            pos_label = 10,
            average = 'weighted'
            )
            rec_scores.append(rec)

            f1 = f1_score(
                y_test,
                prediction,
                pos_label = 10,
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

# def run_all_models_and_score_k_fold(df):
    
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

    # df_metrics = pd.DataFrame(
    #     columns= [
    #         'Model', 'KFold_Score', 'R2: ',
    #         'MSE', 'RMSE', 'MAE','Accuracy',
    #         'Balanced_Acc', 'Precision:', 'Recall', 'F1_Score'
    #     ]
    # )

    for i, model_n in enumerate(model_names):

        # model_metrics_dict = {}

        metrics_i = k_fold_score_new(df, model_names[i])

        # if i < 4:

        st.write(model_n)
        st.write('')
        st.write('kfold score: ',metrics_i[0]*100,'%') #prints score kfold
        st.write('')

        for j, name in enumerate(metrics_names):
            st.write(name, metrics_i[j-1]) #r2, mse, rmse, mae, acc, bacc, prec, rec, f1

            st.write('')

        # else:

        #     st.write(model_n)
        #     st.write('')
        #     st.write('kfold score: ',metrics_i[0]*100,'%') #prints score kfold
        #     st.write('')
            
        #     for j, name in enumerate(metrics_names):
        #           st.write(name, metrics_i[j-1]) #r2, mse, rmse, mae, acc, bacc, prec, rec, f1
        #     st.write('')

# --- New Added ---

# def kfold_cross_validation(df, k = 10, target = 'category'): 
    # """ 
    # K_Fold:
    
    #     - k at 10 has been found through epxperimentation to generally 
    #         result in a model skill estimate with loaw bias and a modest variance.

    # Cross-Validation:

    # - Train/Test Split: Taken to one extreme, k may be set to 2 (not 1) such that a single
    #     train/test split is created to evaluate the model.

    # - LOOCV: Taken to another extreme, k may be set to the total number of observations
    #     in the dataset such that each observation is given a chance to be the held out of the dataset.
    #     This is called leave-one-out cross-validation, or LOOCV for short.

    # - Stratified: The splitting of data into folds may be governed by criteria such as
    #     ensuring that each fold has the same proportion of observations with a given categorical value,
    #     such as the class outcome value. This is called stratified cross-validation.

    # - Repeated: This is where the k-fold cross-validation procedure is repeated n times,
    #     where importantly, the data sample is shuffled prior to each repetition,
    #     which results in a different split of the sample.

    #  - Nested: This is where k-fold cross-validation is performed within each fold of cross-validation,
    #     often to perform hyperparameter tuning during model evaluation. 
    #     This is called nested cross-validation or double cross-validation.
    # """
    # # --- KFOLD & DATA ---

    # kfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = 42)
    # features = df[[col for col in df if col != target]]
    # target = df[target]   

    # # --- CREATING MODELS ---

    # def create_models():
    #     models = list()

    #     models.append(LogisticRegression(solver = 'lbfgs'))
    #     models.append(KNeighborsClassifier(n_neighbors = 3)) # k = 5 by default
    #     models.append(MultinomialNB())
    #     models.append(RandomForestClassifier(max_depth=10, random_state=42))

    #     return models

    # # --- CROSS_VALIDATION ---

    # def cv_eval_model(model, cv, X, y):

    #     scoring = {
    #         'R2': make_scorer(r2_score, average = 'samples'),
    #         'MSE': make_scorer(mean_squared_error, average = 'samples'),
    #         'MAE': make_scorer(mean_absolute_error, average = 'samples'),
    #         'Accuracy': make_scorer(accuracy_score, average = 'samples'),
    #         'Balanced_Acc': make_scorer(balanced_accuracy_score, average = 'samples'),
    #         'Precision': make_scorer(precision_score, average = 'samples'),
    #         'Recall': make_scorer(recall_score, average = 'samples'),
    #         'F1': make_scorer(f1_score, average = 'samples'),
    #     }

    #     cross_val_obj = cross_validate(model, X, y, scoring = scoring, cv = cv, n_jobs = -1, return_estimator = True)

    #     trained_model = cross_val_obj['estimator'][-1]

    #     return trained_model, cross_val_obj

    # # --- RUNNING CROSS-VALIDATION ---

    # def run_cross_val(models, cv = kfold, X = features, y = target):
    #     for model in models:
    #         trained_model, cv_mean = cv_eval_model(model, cv, X, y)

    #         with open (model.__name__, 'wb') as f:
    #             pickle.dump(model,f) 

def kfold_cross_validation(df, k = 10, target = 'category'): 
    # --- KFOLD & DATA ---

    kfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = 42)
    features = df[[col for col in df if col != target]]
    target = df[target]   

    # --- CREATING MODELS ---

    def create_models():
        models = list()

        models.append(LogisticRegression(solver = 'lbfgs'))
        models.append(KNeighborsClassifier(n_neighbors = 3)) # k = 5 by default
        models.append(MultinomialNB())
        models.append(RandomForestClassifier(max_depth=10, random_state=42))

        return models

    # --- CROSS_VALIDATION ---

    def cv_eval_model(model, cv, X, y):

        scoring = {
            'R2': make_scorer(r2_score),
            'MSE': make_scorer(mean_squared_error),
            'MAE': make_scorer(mean_absolute_error),
            'Accuracy': make_scorer(accuracy_score),
            'Balanced_Acc': make_scorer(balanced_accuracy_score),
            'Precision': make_scorer(precision_score, average = 'macro'),
            'Recall': make_scorer(recall_score, average = 'macro'),
            'F1': make_scorer(f1_score, average = 'macro'),
        }

        cross_val_obj = cross_validate(model, X, y, scoring = scoring, cv = cv, n_jobs = -1, return_estimator = True)  

        return cross_val_obj # trained_model, 

    # --- RUNNING CROSS-VALIDATION ---

    def run_cross_val(models, cv = kfold, X = features, y = target):
        dict_df = dict()

        for model in models:
            cv_mean = cv_eval_model(model, cv, X, y) #trained_model, 

            # with open (type(model).__name__, 'wb') as f:
            #     pickle.dump(model,f) 

            dict_df[type(model).__name__] = cv_mean

        return dict_df

    models_ = create_models()
    dict_df = run_cross_val(models = models_)
    return dict_df
    
def best_model(df, k = 10, target = 'category'):
    # --- KFOLD & DATA ---

    kfold = StratifiedKFold(n_splits = k, shuffle = True, random_state = 42)
    features = df[[col for col in df if col != target]]
    target = df[target]  

    # --- CREATE MODELS ---

    def create_models():
        models = list()

        models.append(LogisticRegression(solver = 'lbfgs'))
        models.append(KNeighborsClassifier()) # k = 5 by default
        models.append(MultinomialNB())
        models.append(RandomForestClassifier(random_state=42))

        return models

    # --- SCORES ---

    scoring = {
    'R2': make_scorer(r2_score),
    'MSE': make_scorer(mean_squared_error),
    'MAE': make_scorer(mean_absolute_error),
    'Accuracy': make_scorer(accuracy_score),
    'Balanced_Acc': make_scorer(balanced_accuracy_score),
    'Precision': make_scorer(precision_score, average = 'macro'),
    'Recall': make_scorer(recall_score, average = 'macro'),
    'F1': make_scorer(f1_score, average = 'macro'),
    }

    # --- PARAMETERS ---
        # LR
    Cs = [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 100]

        # KNN
    n_neighbors_ = [3, 5, 8]
        # MultiNB
    alphas = [0, 0.5, 1.0]
        # RandomForest
    n_estimators_ = [50, 100, 150]
    max_features_ = ['sqrt', 'log2']
    ccp_alphas = [0, 0.5, 1]

    dict_param_grid = {
        'LogisticRegression': {'C': Cs},
        'KNeighborsClassifier': {'n_neighbors': n_neighbors_},
        'MultinomialNB': {'alpha': alphas},
        'RandomForestClassifier': {'n_estimators': n_estimators_, 'max_features': max_features_, 'ccp_alpha': ccp_alphas}
    }

    # --- RUNNING GRIDSEARCHCV
    models = create_models()

    df_models = pd.DataFrame(columns = ['model_name', 'best_params', 'best_score', 'best_model'])

    # --- PROGRESS BAR ---
    my_bar = st.progress(0)

    for i, model in enumerate(models):
        param_grid = dict_param_grid[type(model).__name__]
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv = kfold,
            scoring = scoring,
            refit = 'Accuracy',
            n_jobs = -1
        )
        grid_search.fit(features, target)

        df_input = pd.DataFrame({'model_name':[f'{type(model).__name__}'], 'best_params':[grid_search.best_params_], 'best_score':[grid_search.best_score_], 'best_model': [grid_search.best_estimator_]})
        df_models = pd.concat([df_models,df_input], axis = 0, sort = False).reset_index(drop = True)

        my_bar.progress((i+1)/len(models))

    return df_models
    