import pandas as pd 
import numpy as np

COLUMNS_TO_REMOVE = [
    'payment_type',
    'velocity_6h', 'velocity_24h', 'velocity_4w',
    'bank_branch_count_8w',
    'date_of_birth_distinct_emails_4w',
    'employment_status',
    'credit_risk_score',
    'email_is_free',
    'housing_status',
    'phone_home_valid',
    'phone_mobile_valid',
    'bank_months_count',
    'has_other_cards',
    'proposed_credit_limit',
    'foreign_request',
    'source',
    'session_length_in_minutes',
    'device_os',
    'keep_alive_session',
    'device_distinct_emails_8w',
    'device_fraud_count',
    'month'
]

def load_dataset(path):
    df = pd.read_csv(path)
    df = df.drop(columns=COLUMNS_TO_REMOVE)
    return df 

def preprocess(df):
    df.fillna(-1, inplace=True)
    # Add diff between prev address month and current address month
    df['address_diff'] = df['current_address_months_count'] - df['prev_address_months_count']
    # Drop categorical columns
    # df = df.drop(columns=['housing_status', 'device_os', 'source', 'payment_type', 'employment_status'])
    return df

def get_features_targets(df):
    target_col = 'fraud_bool'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
