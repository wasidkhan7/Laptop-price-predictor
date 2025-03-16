import streamlit as st
import pickle 
import pandas as pd
import numpy as np

df=pickle.load(open('df.pkl','rb'))
model=pickle.load(open('pipeline.pkl','rb'))


st.title("Laptop Price Predictor")

# User inputs
brand = st.selectbox("Brand", df['Company'].unique())
type = st.selectbox("Type", df['TypeName'].unique())
ram = st.number_input("RAM (GB)", min_value=2, max_value=64, step=2)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS Panel", ["No", "Yes"])
ppi = st.number_input("Pixel Density (PPI)", min_value=80.0, max_value=300.0, step=0.1)
processor = st.selectbox("Cpu Brand", df['Cpu brand'].unique())
ssd = st.selectbox("SSD (GB)",[0,64,128,512,1024,2048])
hdd = st.selectbox("HDD (GB)", [0,64,128,512,1024,2048])
gpu = st.selectbox("GPU Brand", df['Gpu brand'].unique())
os = st.selectbox("Operating System", df['Os'].unique())

# Convert categorical to numerical
touchscreen = 1 if touchscreen == "Yes" else 0
ips = 1 if ips == "Yes" else 0

# Create input query
querry = np.array([[brand, type, ram, weight, touchscreen, ips, ppi, processor, ssd, hdd, gpu, os]])
querry_df = pd.DataFrame(querry, columns=['Company','TypeName','Ram','Weight','Touchscreen','IPSpanel','PPI','Cpu brand','SSD','HDD','Gpu brand','Os'])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(querry_df)[0]
    exp_pred=np.exp(prediction)
    st.success(f"Estimated Laptop Price: IND Rs{exp_pred:.2f}")



