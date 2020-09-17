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

from PIL import Image

# Streamlit Cache
@st.cache(suppress_st_warning=True)

def main_pipe(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

def scrappe_pipe(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

def upload_pipe(obj, *fns):
    return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

# def pipe(obj, *fns):
#     return functools.reduce(lambda x, y: y(x), [obj] + list(fns))

def hist_of_target_creator(df, target = 'category'):
 
    fig = px.histogram(df, x = target, color= target) #, marginal="rug", hover_data=tips.columns)

    # Plot!
    st.plotly_chart(fig)

    return df

def streamlit_pipe_write_before(df_):
    st.write(f"**Before Over-Under-Sampling")
    return(df_)
    
def streamlit_pipe_write_after(df_):
    st.write(f"**After Over-Under-Sampling**")
    return(df_)

def streamlit_pipe_write_paragraph(url):
    st.write(f"# Supervised Machine Learning Algorithms Classifier,")
    st.write(f"where we pull data from the website 'https://clientsfromhell.net/'")
    st.write(f"Once we have the data, we go case by case in each category and only pull")
    st.write(f"any dialogues, specifically what the client would say per case.")
    st.write(f"the rest of the process can be explained by the line of code above.")
    return(url)

def streamlit_pipe_write_intro():
    st.markdown("<h1 style='text-align: center; color: black;'>Supervised Machine Learning</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: black;'>Algorithms Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'>the scrapped website: <a class='cfh-link' href='https://clientsfromhell.net/'>Clients From Hell</a></h1>", unsafe_allow_html=True) 
    path_cfh_logo = "cfh_logo.png"
    cfh_logo = Image.open(path_cfh_logo)
    st.image(cfh_logo, width = 400)
    st.write(f"")
    st.markdown(
        "<p style='text-align: center;'>Each post made on the website is a story from freelancers.</p>",
        unsafe_allow_html=True
        )
    st.markdown(
        "<p style='text-align: center;'>These stories are of bad experiences the freelancers had with clients<br><strong>in real life...</strong> and each story is classified as the following:</p>",
        unsafe_allow_html=True
        )
    st.markdown(
        "<p style='text-align: center;'><strong>Interactive Graphs: </strong><br>- Click on the categories on the right menu to filter<br>- Hover over columns to see details<br>- Download image graph as png by clicking the camera icon on the image's hover menu</p>",
        unsafe_allow_html=True
        )

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

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

path = "web_logo.png"
image = Image.open(path)

st.image(image, width = 100)

# START SIDE BAR
use_current_data = st.sidebar.selectbox("Re-scrappe website?", ("No", "Yes"))

# Intro
streamlit_pipe_write_intro()

if use_current_data == "No":

    if os.path.isfile('data/data.csv'):
        
        main_pipe(
            upload_pipe(
                CP.data_from_csv('data/data.csv'),
                hist_of_target_creator,
                CP.catetory_replacer,
            )[0],
            streamlit_pipe_write_before,
            hist_of_target_creator,
            CP.over_under_sampling,
            streamlit_pipe_write_after,hist_of_target_creator,
            CP.convert_to_tfidf,
            RM.run_all_models_and_score_k_fold
        )

    else:

        st.sidebar.write('Although "No" was selected, there was no previous data thus scrapping data from website')

        main_pipe(
            scrappe_pipe(
                "https://clientsfromhell.net/",
                S.get_categories,
                S.url_categroy_creator,
                S.page_num_creator,
                S.initialize_scraping,
                CP.df_creator,
                CP.cleaning,
                hist_of_target_creator, 
                CP.data_to_csv,
                CP.catetory_replacer
                )[0],
            hist_of_target_creator,
            streamlit_pipe_write_before,
            hist_of_target_creator,
            CP.over_under_sampling,
            streamlit_pipe_write_after,hist_of_target_creator,
            CP.convert_to_tfidf,
            RM.run_all_models_and_score_k_fold
        )

else:

    try:
        move_old('data')
    except: 
        pass

    main_pipe(
                scrappe_pipe(
                    "https://clientsfromhell.net/",
                    S.get_categories,
                    S.url_categroy_creator,
                    S.page_num_creator,
                    S.initialize_scraping,
                    CP.df_creator,
                    CP.cleaning,
                    hist_of_target_creator,
                    CP.data_to_csv,
                    CP.catetory_replacer,
                    )[0],
                hist_of_target_creator,
                streamlit_pipe_write_before,
                hist_of_target_creator,
                CP.over_under_sampling,
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