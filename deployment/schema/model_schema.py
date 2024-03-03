from pydantic import BaseModel


class TirePerformanceInput(BaseModel):

    tire_width: float
    aspect_ratio: float
    rim_diameter_inches: float
    load_index: float
    material_hardness: float
    tensile_strength: float
    tire_type: str
    construction: str
    speed_rating: str
    season: str