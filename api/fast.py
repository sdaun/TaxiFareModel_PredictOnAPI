from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz
from predict import get_model


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
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict(pickup_datetime, pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude, passenger_count):
    # create a datetime object from the user provided datetime
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")

    # localize the user datetime with NYC timezone
    eastern = pytz.timezone("America/New_York")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # localize the datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    # format pickup datetime to what is expected by pipeline
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    # create dataframe for prediction
    input = {'key': ["2022-05-26 13:03:00.000000119"],
            'pickup_datetime': [formatted_pickup_datetime],
            'pickup_longitude': [float(pickup_longitude)],
            'pickup_latitude': [float(pickup_latitude)],
            'dropoff_longitude': [float(dropoff_longitude)],
            'dropoff_latitude': [float(dropoff_latitude)],
            'passenger_count': [int(passenger_count)]}
    X_pred = pd.DataFrame(input)

    # load model from trained model.joblib
    pipeline = get_model('model.joblib')
    result = pipeline.predict(X_pred)
    return {'fare': float(result[0])}
