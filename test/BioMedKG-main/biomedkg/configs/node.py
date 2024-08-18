from typing import Tuple, Type
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


class NodeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    PRETRAINED_NODE_DIM: int
    GCL_TRAINED_NODE_DIM: int
    KGE_TRAINED_NODE_DIM: int
    MODALITY_TRANSFORM_METHOD : str
    MODALITY_MERGING_METHOD : str

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

node_settings = NodeSettings()