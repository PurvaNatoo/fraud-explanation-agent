import pandas as pd 
import numpy as np

def load_dataset(path):
    df = pd.read_csv(path)
    return df 

def preprocess(df):
    df.fillna(-1, inplace=True)
    # Add diff between prev address month and current address month
    df['address_diff'] = df['current_address_months_count'] - df['prev_address_months_count']
    # Drop categorical columns
    df = df.drop(columns=['housing_status', 'device_os', 'source', 'payment_type', 'employment_status'])
    return df

def get_features_targets(df):
    target_col = 'fraud_bool'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
