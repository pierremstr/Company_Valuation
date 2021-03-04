import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
                      'located', 'names'])
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
    df_copy['clean_info'] = df_copy['description'].apply(clean_info)
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