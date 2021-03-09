import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import streamlit as st



def cleaner(info):
    # Remove punctuation
    for p in string.punctuation:
        info = str(info).replace(p, '')
    # Lower case
    info = info.lower()
    # Remove numbers
    info = ''.join(word for word in info if not word.isdigit())
    # Add common and useless words to stop_words
    stop_words = set(stopwords.words('english'))
    stop_words.update(['company', 'provides', 'offers', 'operates', 'well', 'segment', 'also', 'limited',
                      'headquartered','founded', 'inc', 'management', 'sells', 'including', 'united',
                      'segments', 'states', 'markets', 'various', 'engages', 'addition',
                      'based', 'name', 'business', 'customers', 'formerly', 'known', 'corporation',
                      'subsidiaries', 'group', 'changed', 'develops','approximately','primarily',
                      'related','care','used', 'use', 'include','serves', 'incorporated', 'holdings',
                      'together', 'companys','distributes', 'comprising', 'produces', 'support', 'two',
                      'companies','sales', 'operations', 'ltd','involved','industry','subsidiary', 'owns',
                      'sale', 'three', 'range', 'holding', 'businesses', 'firm', 'product', 'plc',
                      'located', 'names', 'sector', 'country', 'sectors'])
    # Remove stop words
    word_tokens = word_tokenize(info)
    info = [w for w in word_tokens if not w in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in info]
    info = ' '.join(lemmatized)
    return info

def vectorize(df, vectorizer='tfidf', context=2, max_df=0.85, min_df=0.05):
    df_copy = df.copy()
    # Clean language columns
    df_copy['clean_info'] = df_copy['description'].apply(cleaner)
    # Vectorize
    if vectorizer == 'count':
        vectorizer = CountVectorizer(ngram_range=(1,context), max_df=max_df, min_df=min_df)
    if vectorizer == 'tfidf':
        vectorizer = TfidfVectorizer(ngram_range=(1,context), max_df=max_df, min_df=min_df)
    X = vectorizer.fit_transform(df_copy['clean_info'])
    # Convert back to df
    vect_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names(), index= df_copy.index)
    merged_df = df_copy.merge(vect_df, left_index=True, right_index=True, how='left')
    merged_df.drop(columns=['description','clean_info'], inplace=True)
    return merged_df


def get_revenue_size(value):
    if value <= 150:
        return 1
    if 150 < value <= 550:
        return 2
    if 550 < value <= 2500:
        return 3
    else:
        return 4

def transfer_ebitda_margin(value):
    if value >= 0.3:
        return 5
    if 0.2 <= value < 0.3:
        return 4
    if 0.13 <= value < 0.2:
        return 3
    if 0.07 <= value < 0.13:
        return 2
    else:
        return 1

def transfer_growth_rate(value):
    if value >= 0.1:
        return 4
    if 0.05 <= value < 0.1:
        return 3
    if 0 <= value < 0.05:
        return 2
    else:
        return 1

def transfer_roce(value):
    if value >= 0.1:
        return 4
    if 0.05 <= value < 0.1:
        return 3
    if 0.01 <= value < 0.05:
        return 2
    else:
        return 1

def error_pc(y_true, y_pred):
    errors = abs(y_true - y_pred)
    pc_errors = errors/y_true
    return pc_errors.mean()


