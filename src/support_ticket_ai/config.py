"""Validated application settings; credentials are never serialized into prompts."""

from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_path: Path = Path("data/support_tickets.csv")
    data_timezone: str = "UTC"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"
    provider_timeout: float = Field(default=15, gt=0, le=60)
    request_timeout: float = Field(default=60, gt=0, le=120)

    @field_validator("data_timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @property
    def llm_configured(self) -> bool:
        return bool(self.groq_api_key and self.groq_api_key.get_secret_value().strip())
