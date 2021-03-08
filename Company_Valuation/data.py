from sklearn.model_selection import train_test_split
import pandas as pd
import os
from Company_Valuation.utils import transfer_roce, transfer_growth_rate, transfer_ebitda_margin, get_revenue_size

def get_data():
    mydir = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(mydir,'clean_data', 'clean_clean_data.csv')
    df =  pd.read_csv(data_path, index_col=0)
    df = df.drop(columns=["size","symbol"])
    return df

def clean_data(df):
    sector_map = {'Industrials': 'Industrials', 'Technology': 'Information Technology', 'Consumer Cyclical': 'Consumer',
              'Energy': 'Energy', 'Healthcare': 'Healthcare', 'Basic Materials': 'Materials', 'Utilities': 'Utilities',
              'Consumer Defensive': 'Consumer', 'Communication Services': 'Communication Services', 'Metals & Mining': 'Materials',
              'Chemicals': 'Industrials', 'Construction': 'Industrials', 'Textiles, Apparel & Luxury Goods': 'Consumer',
              'Machinery': 'Industrials', 'Electrical Equipment': 'Industrials', 'Media': 'Communication Services',
              'Food Products': 'Consumer', 'Auto Components': 'Industrials', 'Pharmaceuticals': 'Healthcare',
              'Retail': 'Consumer', 'Hotels, Restaurants & Leisure': 'Consumer', 'Consumer products': 'Consumer',
              'Commercial Services & Supplies': 'Materials', 'Trading Companies & Distributors': 'Communication Services',
              'Telecommunication':'Communication Services','Paper & Forest': 'Materials','Industrial Conglomerates': 'Industrials',
              'Transportation Infrastructure': 'Utilities', 'Packaging': 'Industrials',
              'Professional Services': 'Information Technology', 'Beverages': 'Consumer','Semiconductors': 'Materials'}
    country_map = {'US': 'NA', 'CA': 'NA', 'IN': 'EM', 'DE': 'EU', 'HK': 'ROW', 'FR': 'EU', 'GB': 'EU', 'CN':'EM', 'AU': 'ROW',
                   'RU': 'EM','CH':'EU','NL':'EU', 'IE':'EU', 'BE':'EU', 'IL':'EM', 'PT':'EU','BM':'ROW', 'LU':'EU'}
    df = df.sort_values('revenue').head(len(df)-100)  
    df['sector'] = df['sector'].map(sector_map)
    df['country'] = df['country'].map(country_map)
    df['growth_rate'] = df['growth_rate'].apply(transfer_growth_rate)
    df['ebitda_margin'] = df['ebitda_margin'].apply(transfer_ebitda_margin)
    df = df[df['enterpriseValue'] < 10000]
    df = df[((df['enterpriseValue'] / df['ebitda']) > 5) & ((df['enterpriseValue'] / df['ebitda']) < 23)]
    return df

def holdout(df):
    y = df["enterpriseValue"]
    X = df.drop(columns=["enterpriseValue"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test
