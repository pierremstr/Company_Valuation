from sklearn.preprocessing import RobustScaler, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def make_pipeline(model, scaler, num_cols, cat_cols):
    num_transformer = Pipeline([('imputer', SimpleImputer()),
                                ('scaler', scaler)])
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num_transformer', num_transformer, num_cols),
         ('cat_transformer', OneHotEncoder(), cat_cols)],
        remainder='passthrough')
    # Combine preprocessor and model in pipeline
    final_pipe = Pipeline([
        ('preprocessing', preprocessor),
        (f'{model}', model)])
    return final_pipe
