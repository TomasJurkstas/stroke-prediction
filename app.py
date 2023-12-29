import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class HealthcareApplication(BaseModel):
    age: float
    avg_glucose_level: float


class PredictionOut(BaseModel):
    response: float


with open("logisticregression.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def read_root():
    return FileResponse("static/index.html")


@app.post("/predict", response_model=PredictionOut)
def predict(data: HealthcareApplication):
    data_df = pd.DataFrame(data.model_dump(), index=[0])
    prediction = model.predict(data_df)
    result = {'response': prediction}
    return result


@app.post("/predict_probability", response_model=PredictionOut)
def predict(data: HealthcareApplication):
    data_df = pd.DataFrame(data.model_dump(), index=[0])
    prediction = model.predict_proba(data_df)
    result = {'response': prediction[:, 1]}
    return result


app.mount("/", StaticFiles(directory="static"), name="static")
