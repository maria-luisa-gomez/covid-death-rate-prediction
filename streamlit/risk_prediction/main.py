import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
# from streamlit_player import st_player
from sklearn.ensemble import RandomForestClassifier
from src.support import model
import time
import base64




# st.write("""
# # Simple Iris Flower Prediction App
# This app predicts the **Iris flower** type!
# """)



# st_player("https://www.youtube.com/watch?v=Keaa4hOWnzU?autoplay=1")


# st.write("""
# # COVID-19 patients 
# # **Severity Risk Predictor**
# """)

imagen = Image.open("images/severity_risk.png")
st.image(imagen, use_column_width=True)


uploaded_file = st.file_uploader("Insert pacients profile", type = ['csv'])
if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file)
    df1.to_csv('probando.csv', index = False)
    
    prob_pred = model()
 
    st.subheader('Prediction Probability')
    st.table(prob_pred)


file_ = open("images/virus3.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)
data = "images/vaccine_animation.mp4"
st.video(data, format="video/mp4")