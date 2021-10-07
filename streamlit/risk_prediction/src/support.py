import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import pickle



def model():
    
    df = pd.read_csv('probando.csv')
    xgb_model = pickle.load(open('final_model.pkl','rb'))
    scaler = pickle.load(open("./scaler.pkl", "rb"))
    columns = ["Probabilidad_de_caso_LEVE", "Probabilidad_de_caso_GRAVE"]
    
    df = scaler.transform(df)
    
    # predictions proba
    probas = xgb_model.predict_proba(df)
    df_final = pd.DataFrame(probas, columns=columns)
    
    # predictions 0 1
    dict_pred = {
        0 : "LEVE", 
        1 : "GRAVE", 
    }
    predictions = pd.Series(xgb_model.predict(df))
    pred = predictions.map(dict_pred)
    
    df_final["Pronóstico"] = pred
    
    return df_final.sort_values(by = ["Probabilidad_de_caso_GRAVE"], ascending=False)