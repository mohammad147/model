from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the model from the joblib file
def load_model(file_path):
    model = joblib.load(file_path)
    return model

# Load the trained model
model = load_model('random_forest_regressor.pkl')

# Initialize the FastAPI app
app = FastAPI()

# Define the request body model
class PredictionRequest(BaseModel):
    age: int
    height: int
    weight: int
    place_ids: list

# Define the prediction endpoint
@app.post("/predict/")
async def predict(request: PredictionRequest):
    new_data = {
        'Age': request.age,
        'Height': request.height,
        'Weight': request.weight
    }
    predicted_ratings = {}

    for place_id in request.place_ids:
        new_data['Place id'] = place_id
        new_df = pd.DataFrame([new_data])
        predicted_rating = model.predict(new_df)
        predicted_ratings[place_id] = predicted_rating[0]

    return {"predicted_ratings": predicted_ratings}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
