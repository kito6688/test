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
#rfr_models = joblib.load("rfr.joblib")
#lgbm_model = lgb.Booster(model_file="lg.txt")

# Create a sidebar for user input
with st.sidebar:
    st.subheader('Car Specs to Predict Price')
make_model = st.sidebar.selectbox("Model Selection",  
                                  ("Linear", 
                                   "Decision Tree", 
                                   "Random forest", 
                                   "LightGBM"))
#st.sidebar.title("Used Car Price Prediction")
#car_make = st.sidebar.selectbox("Car Make", df['Car Make'].unique())
car_make = st.sidebar.selectbox("Car Make", ('TOYOTA','HONDA','ISUZU','HYUNDAI','KIA'))
if car_make == "TOYOTA":
    car_model = st.sidebar.selectbox("Car Model", 
                                     ('VIGO 4x4',
                                      'VIGO 4x2',
                                      'VIGO CHAMP 4x2',
                                      'VIGO CHAMP PRERUNNER',
                                      'VIGO CHAMP 4x4',
                                      'REVO 4x4',
                                      'REVO PRERUNNER',
                                      'REVO SMART CAB',
                                      'RUSH',
                                      'SIENNA',
                                      'TUNDRA',
                                      'CAMRY',
                                      'COROLLA AKTIS',
                                      'COROLLA CROSS',
                                      'VIOS',
                                      'YARIS',
                                      'GT 86',
                                      'HIACE',
                                      'LAND CRUISER',
                                      'PRADO',
                                      'FJ CRUISER',
                                      'FORTUNER'))
    car_make_v= 5
elif car_make == "HONDA":
    car_model = st.sidebar.selectbox("Car Model", 
                                     ('CR-V',
                                      'HR-V',
                                      'CIVIC',
                                      'CITY',
                                      'ACCOST'))
    car_make_v= 0
elif car_make == "ISUZU":
    car_model = st.sidebar.selectbox("Car Model", 
                                     ('DEMAX',
                                      'Demax-4x2'))
    car_make_v= 2
elif car_make == "KIA":
    car_model = st.sidebar.selectbox("Car Model", 
                                     ('Seltos',
                                      'Cerato',
                                      'Sportage',
                                      'picanto',
                                      'Rio'))
    car_make_v= 3
    if car_model =="Seltos":
        Car_model_v=35
    elif car_model == "Cerato":
        Car_model_v=8
    elif car_model == "Sportage":
        Car_model_v=36
    elif car_model == "picanto":
        Car_model_v=23
    elif car_model == "Rio":
        Car_model_v=30
else:
    car_model = st.sidebar.selectbox("Car Model", 
                                     ('ELANTA',
                                      'ACCENT',
                                      'I30',
                                      'Grant-i',
                                      'SETA'))
    car_make_v= 1
location=st.sidebar.selectbox("Location",
                               ("VTE",
                                "SVK",
                                "HP",
                                "CPS",
                                "XK",
                                "CM",
                                "BK",
                                "LNT",
                                "LPB",
                                "UDX",
                                "PSR",
                                "SK",
                                "SLV",
                                "XYB",
                                "XSB"))
if location =="VTE":
    locations=3
elif location =="SVK":
    locations=3
else:
    locations=0


steering = st.sidebar.radio(
    "Steering wheel",
    ["Left","Right"],
    horizontal=True,
    index=0,)
if steering == 'Left':
    steerings=0
else:
    steerings=1


fuel = st.sidebar.radio(
    "Fuel",["Benzine","Diesel","Hybrid","EV",'PHEV'],
    horizontal=True,
    index=0)
if fuel == 'Benzine':
    fuels=0
elif fuel == 'Diesel':
    fuels=1
elif fuel == 'Hybrid':
    fuels=2
elif fuel == 'EV':
    fuels=2,
else:
    fuels=2


Condition=st.sidebar.selectbox("Condition",("A","B","C","D","F"))
if Condition=="A":
    Conditions=0
elif Condition=="B":
    Conditions=1
elif Condition=="C":
    Conditions=2
elif Condition == "D":
    Conditions=2
else:
    Condition=2


Drive=st.sidebar.selectbox("Drive",("RWD","FWD","4WD","4x4","4x2"))
if Drive=="RWD":
    Drives=0
elif Drive =="4WD":
    Drives=1
elif Drive =="4x2":
    Drives=1
elif Drive =="4x4":
    Drives=1
else:
    Drives=1

Transmission=st.sidebar.radio("Transmission",["Automatic","Manual"],
                              horizontal=True,
                              index=0)
if Transmission =="Automatic":
    Transmissions=1
else:
    Transmissions=2


Type=st.sidebar.selectbox("Type",("SUV's",
                                  "SUV's",
                                  "suv",
                                  "sedan",
                                  "hatchback",
                                  "Van",
                                  "PPV"))
color=st.sidebar.selectbox("Paint-color",
                           ("white",
                           "BLACK",
                           "BLUE",
                           "YELLOW",
                           "GRAY",
                           "SILVER",
                           "Cream",
                           "Sugar",
                           "Orange",
                           "gold silver",
                           "Red",
                           "Green"))
Cylinders=st.sidebar.radio("Cylinders",["3","4","6","8"],horizontal=True,index=1)
Year = st.sidebar.number_input("Year:",min_value=2000, max_value=2023, value=2010, step=1)
Mileage = st.sidebar.number_input("Mileage (km):",min_value=0, max_value=317000, value=10000, step=5000)

# Add input fields for the other 10 features

my_dict = {"Car make":car_make, "car_model":car_model,"Location":locations,"Steering wheel":steering}
df = pd.DataFrame.from_dict([my_dict])

cols = {
    "car_make": "car_make",
    "car_model": "car_model",
    "Location":"Location",
    "Steering wheel":"Steering wheel"
    
}

df_show = df.copy()
df_show.rename(columns = cols, inplace = True)
st.write("Selected Specs: \n")
st.write("Model:",make_model )
st.table(df_show)
# Define a function to make predictions
#@st.cache(allow_output_mutation=True)
def predict_price(model, features):
    return model.predict(features)


new_data = [[4,1,2,3,2021,11,33,1,2,100000,3,9,0]]
lr_prediction = predict_price(lr_model, new_data)
st.write(f"Linear Regression Prediction: {lr_prediction[0]}")
#rfr_prediction = predict_price(rfr_model, new_data)
#rfr_prediction = rfr_models.predict(new_data)
#st.write(f"Decision Tree Regression Prediction: {rfr_prediction[0]}")