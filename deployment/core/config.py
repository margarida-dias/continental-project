from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Config for the project
    """
    API_V1_STR: str = '/deployment/api/v1'
    MODEL_URI: str = '/Users/margarida/PycharmProjects/continental_project/xg-boost-gs-model.json'

    class Config:
        case_sensitive = True


settings = Settings()
