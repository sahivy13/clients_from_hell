import warnings
warnings.filterwarnings("ignore")

import functools
import streamlit as st

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import plotly.figure_factory as ff
import plotly.express as px

import scrapper as S
import clientsfh_preprocessing as CP
import regression_models as RM

import glob
import os
import platform

# Streamlit Cache
@st.cache(suppress_st_warning=True)

def main_pipe(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

def scrappe_pipe(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

# def pipe(obj, *fns):
#     return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

def hist_of_target_creator(df, target = 'category'):
 
    fig = px.histogram(df, x = target) #, color="sex", marginal="rug", hover_data=tips.columns)

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

def move_old(folder):
    def creation_date(path_to_file):
    
        if platform.system() == 'Windows':
            return os.path.getctime(path_to_file)
        else:
            stat = os.stat(path_to_file)
            try:
                return stat.st_birthtime
            except AttributeError:
                # We're probably on Linux. No easy way to get creation dates here,
                # so we'll settle for when its content was last modified.
                return stat.st_mtime

    def move_data():
        os.rename(
        f"{folder}/{folder}.csv",
        f"{folder}/previous_{folder}/{folder}_{creation_date(f'{folder}/{folder}.csv')}.csv"
        )
    
    move_data()

if os.path.isfile('data/data.csv'):
    
    move_old('data')

    if os.path.isfile('gaussian/gaussian'):
    
    else:

        with st.echo():
            main_pipe(
                CP.data_from_csv('data/data.csv'),
                hist_of_target_creator,
                streamlit_pipe_write_before,
                hist_of_target_creator,
                CP.under_sampling_to_2_col_by_index,
                streamlit_pipe_write_after,hist_of_target_creator,
                CP.convert_to_tfidf,
                RM.run_all_models_and_score_k_fold
            )

else:
    with st.echo():

        main_pipe(
            scrappe_pipe(
                "https://clientsfromhell.net/",
                streamlit_pipe_write_paragraph,
                S.get_categories,
                S.url_categroy_creator,
                S.page_num_creator,
                S.initialize_scraping,
                CP.df_creator,
                CP.cleaning,
                hist_of_target_creator, 
                CP.catetory_replacer,
                CP.data_to_csv
                ),
            hist_of_target_creator,
            streamlit_pipe_write_before,
            hist_of_target_creator,
            CP.under_sampling_to_2_col_by_index,
            streamlit_pipe_write_after,hist_of_target_creator,
            CP.convert_to_tfidf,
            RM.run_all_models_and_score_k_fold
        )

# with st.echo():
#     main_pipe(
#         scrappe_pipe(
#             "https://clientsfromhell.net/",
#             streamlit_pipe_write_paragraph,
#             S.get_categories,
#             S.url_categroy_creator,
#             S.page_num_creator,
#             S.initialize_scraping,
#             CP.df_creator,
#             CP.cleaning,
#             hist_of_target_creator, 
#             CP.catetory_replacer,
#             CP.data_to_csv
#             ),
#         streamlit_pipe_write_before,
#         hist_of_target_creator,
#         CP.under_sampling_to_2_col_by_index,
#         streamlit_pipe_write_after,hist_of_target_creator,
#         CP.convert_to_tfidf,
#         RM.run_all_models_and_score_k_fold
#     )