from typing import Optional, Tuple, Type
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


class KGESetting(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    KGE_NODE_INIT_METHOD: str
    KGE_ENCODER : Optional[str]
    KGE_DECODER : str
    KGE_HIDDEN_DIM : int
    KGE_NUM_HIDDEN : int
    KGE_NUM_HEAD : Optional[int] = 1
    KGE_DROP_OUT : bool

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

kge_settings = KGESetting()