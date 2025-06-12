import mlflow
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import requests
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


encoder_Cloud_Cover = joblib.load("model/encoder_Cloud_Cover.joblib")
encoder_Location = joblib.load("model/encoder_Location.joblib")
encoder_Season = joblib.load("model/encoder_Season.joblib")
pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")
scaler_Atmospheric_Pressure = joblib.load("model/scaler_Atmospheric_Pressure.joblib")
scaler_Humidity = joblib.load("model/scaler_Humidity.joblib")
scaler_Precipitation = joblib.load("model/scaler_Precipitation.joblib")
scaler_Temperature = joblib.load("model/scaler_Temperature.joblib")
scaler_UV_Index = joblib.load("model/scaler_UV_Index.joblib")
scaler_Visibility = joblib.load("model/scaler_Visibility.joblib")
scaler_Wind_Speed =joblib.load("model/scaler_Wind_Speed.joblib")

pca_numerical_columns_1 = [
    'Temperature',
    'Humidity',
    'Wind_Speed',
    'UV_Index',
    'Visibility_(km)'
]
 
pca_numerical_columns_2 = [
    'Precipitation_(%)',
    'Atmospheric_Pressure'
]
 
def data_preprocessing(data):
    """Preprocessing data
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the data to make prediction 
        
    return:
        Pandas DataFrame: Dataframe that contain all the preprocessed data
    """
    data = data.copy()
    df = pd.DataFrame()
    
    df["Atmospheric_Pressure"] = scaler_Atmospheric_Pressure.transform(np.asarray(data["Atmospheric_Pressure"]).reshape(-1,1))[0]

    df["Cloud_Cover"] = encoder_Cloud_Cover.transform(data["Cloud_Cover"])[0]
    df["Location"] = encoder_Location.transform(data["Location"])
    df["Season"] = encoder_Season.transform(data["Season"])
    
    # PCA 1
    data["Temperature"] = scaler_Temperature.transform(np.asarray(data["Temperature"]).reshape(-1,1))[0]
    data["Humidity"] = scaler_Humidity.transform(np.asarray(data["Humidity"]).reshape(-1,1))[0]
    data["Wind_Speed"] = scaler_Wind_Speed.transform(np.asarray(data["Wind_Speed"]).reshape(-1,1))[0]
    data["UV_Index"] = scaler_UV_Index.transform(np.asarray(data["UV_Index"]).reshape(-1,1))[0]
    data["Visibility_(km)"] = scaler_Visibility.transform(np.asarray(data["Visibility_(km)"]).reshape(-1,1))[0]

    df[["pc1_1", "pc1_2", "pc1_3", "pc1_4", "pc1_5"]] = pca_1.transform(data[pca_numerical_columns_1])
    
    # PCA 2
    data["Precipitation_(%)"] = scaler_Precipitation.transform(np.asarray(data["Precipitation_(%)"]).reshape(-1,1))[0]
    data["Atmospheric_Pressure"] = scaler_Atmospheric_Pressure.transform(np.asarray(data["Atmospheric_Pressure"]).reshape(-1,1))[0]

    df[["pc2_1", "pc2_2"]] = pca_2.transform(data[pca_numerical_columns_2])
    
    return df

def prediction(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (Good, Standard, or Poor)
    """
    # URL endpoint dari model yang sedang di-serve
    url = "http://127.0.0.1:5005/invocations"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=data, headers=headers)
    response = response.json().get("predictions")
    result_target = joblib.load("model/encoder_target.joblib")
    final_result = result_target.inverse_transform(response)
    return final_result