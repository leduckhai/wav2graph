from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Set, Optional

class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    NODES_LST : Set[str]
    EDGES_LST : Set[str]
    DATA_DIR : str
    BENCHMARK_DIR : Optional[str]

data_settings = DataSettings()