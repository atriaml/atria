from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "atria"

    ATRIAX_URL: str = "http://127.0.0.1:8000"
    ATRIAX_STORAGE_URL: str = "http://127.0.0.1:8090"
    ATRIAX_ANON_KEY: str = "sb_publishable_cY44rYmTawJwU1coFeruPX__momBm12"
    LOG_FORMAT: str = "%(message)s"
    DATE_FORMAT: str = "[%X]"


settings = Settings()  # type: ignore
