# serve_api.py
import joblib
import uvicorn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load trained model
model = joblib.load("artifacts/model.joblib")

app = FastAPI()

class TitanicInput(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int

@app.post("/predict")
def predict(input_data: TitanicInput):
    X = np.array([[input_data.Pclass, input_data.Sex, input_data.Age,
                   input_data.SibSp, input_data.Parch, input_data.Fare,
                   input_data.Embarked]])
    pred = model.predict(X)
    return {"prediction": int(pred[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
