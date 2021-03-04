from Company_Valuation.data import get_data, holdout


class Trainer():

    def __init__(self):
        pass

    def train(self):

        # get data
        df = get_data()



        # hold out
        X_train, X_test, y_train, y_test = holdout(df)

