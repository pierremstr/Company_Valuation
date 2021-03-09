from Company_Valuation.data import get_data, holdout, clean_data, classify_holdout
from Company_Valuation.utils import vectorize, error_pc
from Company_Valuation.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split

def my_custom_metric(y_true, y_pred):
    correct = 0 
    total = 0
    y_true =y_true.values
    for index, value in enumerate(y_pred):
        if value == 1:
            if y_true[index] == 1:
                correct += 1
                total += 1
            else:
                total += 1
    return correct/total

class Trainer():

    def __init__(self):
        self.X_train = 0
        self.X_test = 0
        self.y_train = 0
        self.y_test = 0

    def train(self, model, scaler=RobustScaler(), remove_features=[]):
        # Split cols for column transformer
        num_cols = ['returnOnCapitalEmployed','grossProfit','ebitda', 'growth_rate', 'ebitda_margin']
        cat_cols = ['sector', 'country']
        for feature in remove_features:
            if feature in num_cols:
                num_cols.remove(feature)
            if feature in cat_cols:
                cat_cols.remove(feature)
        # get data
        df = get_data()
        df = clean_data(df, upper, lower)
        length = len(df)
        # Drop removed features
        df = df.drop(columns=remove_features)
        # NLP
        if not 'description' in remove_features:
            df = vectorize(df)
        # Hold out
        self.X_train, self.X_test, self.y_train, self.y_test = holdout(df)
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
        # Make pipe
        pipe = make_pipeline(model, scaler, num_cols, cat_cols)
        if remove_features:
           print(f'Training without {remove_features}')
        mse = abs(cross_val_score(pipe, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error').mean())
        print(f'Cross Validated RMSE = {mse ** 0.5}')
        error_scorer = make_scorer(error_pc, greater_is_better=False)
        error = abs(cross_val_score(pipe, self.X_train, self.y_train, cv=5, scoring=error_scorer).mean())
        print(f'Percentage Error = {error}')
        pipe.fit(self.X_train, self.y_train)
        return pipe
    
    def classify(self, model, scaler=RobustScaler(), remove_features=[]):
        # Split cols for column transformer
        num_cols = ['returnOnCapitalEmployed','grossProfit','ebitda', 'growth_rate', 'ebitda_margin']
        cat_cols = ['sector', 'country']
        for feature in remove_features:
            if feature in num_cols:
                num_cols.remove(feature)
            if feature in cat_cols:
                cat_cols.remove(feature)
        # get data
        df = get_data()
        df = clean_data(df)
        # Drop removed features
        df = df.drop(columns=remove_features)
        # NLP
        if not 'description' in remove_features:
            df = vectorize(df)
        # Hold out
        self.X_train, self.X_test, self.y_train, self.y_test = classify_holdout(df)
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
        # Make pipe
        pipe = make_pipeline(model, scaler, num_cols, cat_cols)
        if remove_features:
           print(f'Training without {remove_features}')
        # mse = abs(cross_val_score(pipe, self.X_train, self.y_train, cv=5, scoring='neg_mean_squared_error').mean())
        # print(f'Cross Validated RMSE = {mse ** 0.5}')
        precision_scorer = make_scorer(my_custom_metric, greater_is_better=True)
        precision = abs(cross_val_score(pipe, self.X_train, self.y_train, cv=5, scoring=precision_scorer).mean())
        print(f'Percentage Correctly Identified in our Range = {precision}')
        pipe.fit(self.X_train, self.y_train)
        return pipe#, self.X_train, self.y_train

    def evaluate(self, pipe):
        y_pred = pipe.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = mse ** 0.5
        return {'mae': mae, 'mse': mse, 'rmse': rmse}

