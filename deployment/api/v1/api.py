from fastapi import APIRouter

from deployment.api.v1.endpoints import tire_performance

api_router = APIRouter()
api_router.include_router(tire_performance.router, prefix='/tire_performance', tags=["tire_performance"])
