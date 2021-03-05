from Company_Valuation.data import get_data, holdout
from Company_Valuation.utils import vectorize
from Company_Valuation.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score


class Trainer():

    def __init__(self):
        self.X_train = 0
        self.X_test = 0
        self.y_train = 0
        self.y_test = 0

    def train(self, model, scaler=RobustScaler(),  cv=True, remove_features=[]):
        # Split cols for column transformer
        num_cols = ['returnOnCapitalEmployed','ebitda','grossProfit', 'growth_rate', 'ebitda_margin']
        cat_cols = ['sector', 'country']
        for feature in remove_features:
            if feature in num_cols:
                num_cols.remove(feature)
            if feature in cat_cols:
                cat_cols.remove(feature)
        # get data
        df = get_data()
        # Drop removed features
        df = df.drop(columns=['symbol'])
        df = df.drop(columns=remove_features)
        # NLP
        if not 'description' in remove_features:
            df = vectorize(df)
        # Hold out
        self.X_train, self.X_test, self.y_train, self.y_test = holdout(df)
        # Make pipe
        pipe = make_pipeline(model, scaler, num_cols, cat_cols)
        if remove_features:
            print(f'Training without {remove_features}')
        if cv:
            cv_score = abs(cross_val_score(pipe, self.X_train, self.y_train, cv=5, scoring='neg_mean_absolute_error').mean())
            print(f'Cross Validated MAE = {cv_score}')
            pipe.fit(self.X_train, self.y_train)
            return pipe
        pipe.fit(self.X_train, self.y_train)
        return pipe

    def evaluate(self, pipe):
        y_pred = pipe.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = mse ** 0.5
        return {'mae': mae, 'mse': mse, 'rmse': rmse}

