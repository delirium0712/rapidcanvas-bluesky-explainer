import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    # LLM (OpenAI-compatible HTTP API)
    openai_api_key: str

    openai_api_base: str = "https://api.openai.com/v1"
    openai_chat_model: str = "gpt-4.1-mini"

    # Bluesky public AppView base
    bluesky_appview_base: str = "https://api.bsky.app"

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables.

        Required:
        - OPENAI_API_KEY

        Optional overrides:
        - OPENAI_API_BASE
        - OPENAI_CHAT_MODEL
        - BLUESKY_APPVIEW_BASE
        """

        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Set it to an OpenAI-compatible API key to run the agent."
            )

        return cls(
            openai_api_key=openai_api_key,
            openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
            bluesky_appview_base=os.getenv(
                "BLUESKY_APPVIEW_BASE", "https://api.bsky.app"
            ),
        )
