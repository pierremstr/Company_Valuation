from sklearn.model_selection import train_test_split
import pandas as pd
import os
from Company_Valuation.utils import transfer_roce, transfer_growth_rate, transfer_ebitda_margin, get_revenue_size, transfer_ev

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
    #df = df[(df['ebitda'] < upper) & (df['ebitda'] > lower)]
    df['sector'] = df['sector'].map(sector_map)
    df['country'] = df['country'].map(country_map)
    df['growth_rate'] = df['growth_rate'].apply(transfer_growth_rate)
    df['ebitda_margin'] = df['ebitda_margin'].apply(transfer_ebitda_margin)
    df = df[((df['enterpriseValue'] / df['ebitda']) > 5) & ((df['enterpriseValue'] / df['ebitda']) < 23)]
    df['ev/ebitda'] = df['enterpriseValue']/ df['ebitda']
    Consumer = df[((df['sector']) == 'Consumer')]
    Industrials = df[((df['sector']) == 'Industrials')]
    Information_Technology = df[((df['sector']) == 'Information Technology')]
    Healthcare = df[((df['sector']) == 'Healthcare')]
    Materials = df[((df['sector']) == 'Materials')]
    Energy = df[((df['sector']) == 'Energy')]
    Utilities = df[((df['sector']) == 'Utilities')]
    Communication_Services = df[((df['sector']) == 'Communication Services')]

    def label_sector (row):
      if row['sector'] == 'Consumer' :
        return Consumer['ev/ebitda'].median()
      if row['sector'] == 'Industrials' :
        return Industrials['ev/ebitda'].median()
      if row['sector'] == 'Information Technology' :
        return Information_Technology['ev/ebitda'].median()
      if row['sector'] == 'Healthcare' :
        return Healthcare['ev/ebitda'].median()
      if row['sector'] == 'Materials' :
        return Materials['ev/ebitda'].median()
      if row['sector'] == 'Energy' :
        return Energy['ev/ebitda'].median()
      if row['sector'] == 'Communication Services' :
        return Communication_Services['ev/ebitda'].median()
      return Utilities ['ev/ebitda'].median()

    df['sector_mutiple'] = df.apply (lambda row: label_sector(row), axis=1)
    df['premium/discount'] = df['ev/ebitda'] / df['sector_mutiple'] -1
    df = df[((df['premium/discount']) > -0.5) & ((df['premium/discount']) < 0.5)]
    df = df[((df['enterpriseValue']) > 300) & ((df['enterpriseValue']) < 15000)]
    df.drop(['ev/ebitda','sector_mutiple','premium/discount'], axis=1, inplace=True)

    return df

def holdout(df):
    y = df["enterpriseValue"]
    X = df.drop(columns=["enterpriseValue"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

def classify_holdout(df):
    df['enterpriseValue'] = df['enterpriseValue'].apply(transfer_ev)
    y = df["enterpriseValue"]
    X = df.drop(columns=["enterpriseValue"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test
