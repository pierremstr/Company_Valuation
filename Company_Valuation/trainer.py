from Company_Valuation.data import get_data, holdout
from Company_Valuation.utils import vectorize
from Company_Valuation.pipeline import make_pipeline
from sklearn.metrics import mae

class Trainer():

    def __init__(self):
        pass

    def train(self, model, scaler,  nlp=True, remove_features=[]):
        num_cols = ['returnOnCapitalEmployed','ebitda','grossProfit', 'growth_rate', 'ebitda_margin']
        cat_cols = ['sector', 'country']
        for feature in remove_features:
            if feature in num_cols:
                num_cols.remove(feature)
            if feature in cat_cols:
                cat_cols.remove(feature)
        # get data
        df = get_data()
        # NLP
        if NLP == False:
            df.drop(columns=['description'], inplace=True)
        else:
            df = vectorize(df)
            cat_cols = ['sector_x', 'country_y']
        # hold out
        X_train, X_test, y_train, y_test = holdout(df)
        # Make pipe
        pipe = make_pipeline(model, scaler, num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        return pipe

    def evaluate(self, pipe):
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        return mae



        
            


