from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. Env vars use the NOMRAG_ prefix (e.g. NOMRAG_PROVIDER);
    API keys keep their canonical names (GEMINI_API_KEY, ANTHROPIC_API_KEY)."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_prefix="NOMRAG_", extra="ignore"
    )

    provider: str = "gemini"  # gemini | anthropic | fake
    # Free-tier quota drives this choice: pro and 2.0-flash return 429 (no quota),
    # 3.5-flash returns 503, and 3.6-flash allows only 20 requests/day — a single
    # eval run needs ~90. The lite tier has real quota. Judge is a different and
    # newer model than the generator, so verdicts are independent (no
    # self-preference bias) and the judge is not weaker than what it grades.
    generation_model: str = "gemini-3.1-flash-lite"
    judge_model: str = "gemini-3.5-flash-lite"
    embedding_model: str = "intfloat/multilingual-e5-small"
    top_k: int = 5

    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / ".chroma"
    traces_dir: Path = PROJECT_ROOT / "traces"

    gemini_api_key: str = Field("", validation_alias="GEMINI_API_KEY")
    anthropic_api_key: str = Field("", validation_alias="ANTHROPIC_API_KEY")

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


settings = Settings()
