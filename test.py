import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import joblib

html_temp = """
<div style="background-color:yellow;padding:1.5px">
<h1 style="color:black;text-align:center;">Used Car Price Prediction</h1>
</div><br>"""
st.markdown(html_temp, unsafe_allow_html=True)

st.write("\n\n"*2)

# Load your dataset
df = pd.read_csv("players1.csv")
#rfr_model= RandomForestRegressor()
# Load your pre-trained models
lr_model =LinearRegression()
lr_model = joblib.load("Lr.pkl")
#dtr_model = joblib.load("dtr.pkl")
rfr_models = joblib.load("rfr.joblib")
#lgbm_model = lgb.Booster(model_file="lg.txt")
