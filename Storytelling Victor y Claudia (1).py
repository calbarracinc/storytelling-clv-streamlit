# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 19:32:59 2025

@author: Victor Garcia & Claudia Albcarracin
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Storytelling", layout="wide")
st.title("Storytelling")

uploaded_file = st.sidebar.file_uploader("Sube el archivo por favor", type = ["csv"])
if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file,sep=";")
    
    df["Customer Lifetime Value"]= df["Customer Lifetime Value"].str.replace(".","",regex = False).astype(float)
    df["Effective To Date"] = pd.to_datetime(df["Effective To Date"],errors="coerce")
    
    st.dataframe(df.head())
    st.write(df.dtypes)
    
    st.write(df.groupby("State")["Customer Lifetime Value"].sum())
    st.write(df["Response"].value_counts(normalize  = True))
    st.write(df.nlargest(10,"Customer Lifetime Value"))
    
    histograma = px.histogram(df, x="Customer Lifetime Value")
    st.plotly_chart(histograma, use_container_width = True)
    
    barras = px.bar(df.groupby("State")["Customer Lifetime Value"].sum().reset_index(),
           x="State", y="Customer Lifetime Value")
    st.plotly_chart(barras,use_container_width = True)
    
    dispersion =px.scatter(df, x="Income", y= "Customer Lifetime Value")
    st.plotly_chart(dispersion, use_container_width = True)
    
    caja = px.box(df, x="Vehicle Class", y= "Customer Lifetime Value")
    st.plotly_chart(caja, use_container_width = True)
    
    lineas = px.line(df.groupby("Effective To Date")["Customer Lifetime Value"].sum()
            .reset_index(),x="Effective To Date", y="Customer Lifetime Value")
    st.plotly_chart(lineas,use_container_width = True)
    
    st.markdown("- Estado con mayor CLV:")
    st.markdown("- Tasa de respuesta: ")
else:
    st.info("Por favor sube el archivo para visualizar el storytelling")
    
#cd "C:/Users/Victor Garcia/Downloads"
#streamlit run "Storytelling Victor Y Claudia.py"

