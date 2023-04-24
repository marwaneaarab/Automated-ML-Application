import streamlit as st
import pandas as pd
import os


# Import profiling capability
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# ML stuff
from pycaret.classification import setup , compare_models, pull, save_model

if os.path.exists("./sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

with st.sidebar:
    st.image("https://miro.medium.com/v2/resize:fit:828/format:webp/1*COn2BVUBRrrZ5HVFMycFmA.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Machine Learning", "Download"])
    st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling and PyCaret.")

if choice == "Upload":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload Your Dataset Here :")
    if file:
        
        df = pd.read_csv(file, index_col=None)
        df.to_csv('sourcedata.csv',index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    profile = ProfileReport(df)
    st_profile_report(profile)

if choice == "Machine Learning":
    st.title("Machine Learning :")
    target = st.selectbox("Select Your Target",df.columns)
    if st.button("Train model"):      
        setup(df,target=target) 
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model,'best_model')    
        
if choice == "Download":
    with open("best_model.pkl","rb") as f:
        st.download_button("Download the Model",f,"Trained_model.pkl")
