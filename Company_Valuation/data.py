from sklearn.model_selection import train_test_split
import pandas as pd
import os
from Company_Valuation.utils import transfer_roce, transfer_growth_rate, transfer_ebitda_margin, get_revenue_size

def get_data():
    mydir = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(mydir, 'clean_data', 'clean_clean_data.csv')
    df = pd.read_csv(data_path, index_col=0)
    df = df.drop(columns=['symbol', 'size'])

    return df

def clean_data(df):
    df['growth_rate'] = df['growth_rate'].apply(transfer_growth_rate)
    df['ebitda_margin'] = df['ebitda_margin'].apply(transfer_ebitda_margin)
    df['returnOnCapitalEmployed'] = df['returnOnCapitalEmployed'].apply(transfer_roce)
    df['revenue'] = df['revenue'].apply(get_revenue_size)
    return df

def holdout(df):
    y = df["enterpriseValue"]
    X = df.drop("enterpriseValue", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test
