from typing import Union, List, Tuple, Type
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource

class TrainSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    VAL_RATIO : float
    TEST_RATIO : float
    BATCH_SIZE : int
    
    LEARNING_RATE : float
    EPOCHS : int
    SCHEDULER_TYPE : str
    WARM_UP_RATIO : float

    DEVICES: Union[int, List[int], str, List[str]]
    SEED : int
    OUT_DIR : str
    LOG_DIR : str
    VAL_EVERY_N_EPOCH : int

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return dotenv_settings, env_settings, init_settings

train_settings = TrainSettings()