from Company_Valuation.data import get_data, holdout
from Company_Valuation.utils import cleaner, vectorize


class Trainer():

    def __init__(self):
        pass

    def train(self):

        # get data
        df = get_data()

        # clean data description column
        df['description'] = df['description'].apply(cleaner)

        # vectorize the description column
        df_merged = vectorize(df)

        # hold out
        # X_train, X_test, y_train, y_test = holdout(df)

        return df_merged

