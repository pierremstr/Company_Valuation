from sklearn.model_selection import train_test_split
import pandas as pd

def get_data():
    df = pd.read_csv('../Company_Valuation/clean_data/clean_clean_data.csv', index_col='Unnamed: 0')
    return df

def clean_data(df):
    pass

def holdout(df):
    y_train = df["enterpriseValue"]
    X_train = df.drop("enterpriseValue", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
    return X_train, X_test, y_train, y_test