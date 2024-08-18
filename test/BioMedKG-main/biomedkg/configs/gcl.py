from typing import Tuple, Type
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


class GCLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    GCL_HIDDEN_DIM : int
    GCL_NUM_HIDDEN : int
    GCL_DROP_OUT : bool

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

gcl_settings = GCLSettings()