import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"ok": "True"}


@app.get("/predict_enterprise_value/")
def create_enterprise_value(revenue,
                            ebitda,
                            net_debt,
                            revenue_growth,
                            return_on_capital_employed,
                            sector,
                            region):

    # revenue in US$m
    # ebitda in US$m 
    # net debt in US$m
    # revenue growth last 3 years
    # return on capital employed in percent (e.g. 15)
    # sector:Consumer, Communication Services, Utilities, Industrials, Materials, Information Technology, Healthcare, Energy
    # region = EM (Emerging Markets), ROW (Rest of the world), NA (North America), EU (European Union)

    # build X ⚠️ beware to the order of the parameters ⚠️
    X = pd.DataFrame(dict(
        revenue=[float(revenue)],
        ebitda=[float(ebitda)],
        net_debt=[float(net_debt)],
        revenue_growth=[float(revenue_growth)],
        return_on_capital_employed=[float(return_on_capital_employed)],
        sector=[sector],
        region=[region]))

    # ⚠️ TODO: get model from GCP

    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(prediction=pred)

