from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends
from starlette import status

from deployment.core.config import settings
from deployment.schema.model_schema import TirePerformanceInput
import xgboost as xgb

"""
example for request:
{
    "tire_width": 145,
    "aspect_ratio": 40,
    "rim_diameter_inches": 14,
    "load_index": 75,
    "material_hardness": 50.0,
    "tensile_strength": 10.0,
    "tire_type": "P",
    "construction": "R",
    "speed_rating": "S",
    "season": "Summer"
}
"""

router = APIRouter()


def load_model() -> xgb.XGBRegressor:
    """
    Loads model
    :return: xgb.XGBRegressor model
    """
    model = xgb.XGBRegressor()
    model.load_model(settings.MODEL_URI)
    return model


@router.post("/predict/", status_code=status.HTTP_200_OK)
async def predict_performance(input_data: TirePerformanceInput,
                              model=Depends(load_model)):
    """
    Predicts the performance of the tire
    :param model
    :param input_data:
    :return: Performance
    """
    processed_data = preprocess_data(input_data)

    prediction = model.predict(np.array([processed_data]))

    prediction_value = prediction[0].item()

    return {"performance_metric": prediction_value}


def preprocess_data(input_data: TirePerformanceInput) -> list:
    """
    Function to preprocess raw data
    :param input_data:
    :return: preprocessed data
    """
    tire_types = ['P', 'LT', 'ST', 'T']
    constructions = ['R', 'D', 'B']
    speed_ratings = ['S', 'T', 'U', 'H', 'V']
    seasons = ['Summer', 'Winter', 'All-Season']

    # Initialize one-hot encoded lists for each categorical variable
    tire_type_encoded = [1 if input_data.tire_type == tt else 0 for tt in tire_types]
    construction_encoded = [1 if input_data.construction == c else 0 for c in constructions]
    speed_rating_encoded = [1 if input_data.speed_rating == sr else 0 for sr in speed_ratings]
    season_encoded = [1 if input_data.season == s else 0 for s in seasons]

    # Combine all features into a single list
    processed_data = [
                         input_data.tire_width,
                         input_data.aspect_ratio,
                         input_data.rim_diameter_inches,
                         input_data.load_index,
                         input_data.material_hardness,
                         input_data.tensile_strength,
                     ] + tire_type_encoded + construction_encoded + speed_rating_encoded + season_encoded

    return processed_data
