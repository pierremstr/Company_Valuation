from sklearn.model_selection import train_test_split
import pandas as pd

def get_data():
    df = pd.read_csv('../Company_Valuation/clean_data/clean_clean_data.csv', index_col='Unnamed: 0')
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
    country_map = {'US': 'US', 'CA': 'US', 'IN': 'EM', 'DE': 'EU', 'HK': 'ROW', 'FR': 'EU', 'GB': 'EU', 'CN':'EM', 'AU': 'ROW',
                   'RU': 'EM','CH':'EU','NL':'EU', 'IE':'EU', 'BE':'EU', 'IL':'EM', 'PT':'EU','BM':'ROW', 'LU':'EU'}
    df['sector'] = df['sector'].apply(sector_map)
    df['country'] = df['country'].apply(country_map)
    return df

def holdout(df):
    y = df["enterpriseValue"]
    X = df.drop("enterpriseValue", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test