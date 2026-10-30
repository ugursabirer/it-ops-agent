"""Central configuration for the IT operations assistant.

Every tunable value is read from the environment (.env) so that no credential,
path or model name is hard-coded in the agent logic. The module exposes a single
``settings`` object plus ready-to-use LLM and embedding clients.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

# FAISS still lives in langchain-community, which emits a sunset warning on
# import. The warning is about the package's future, not about our usage, and it
# would otherwise be printed on top of every agent run.
warnings.filterwarnings(
    "ignore", message=".*langchain-community.*", category=DeprecationWarning
)

BASE_PATH = Path(__file__).resolve().parent

load_dotenv(BASE_PATH / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "evet"}


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else BASE_PATH / path


@dataclass(frozen=True)
class Settings:
    """Runtime settings, resolved once at import time."""

    llm_model: str
    embedding_model: str
    ollama_base_url: str
    llm_temperature: float
    max_output_tokens: int
    use_mcp: bool
    require_approval: bool
    verbose: bool
    db_path: Path
    vector_path: Path
    policy_path: Path

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            llm_model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024")),
            use_mcp=_as_bool(os.getenv("IT_OPS_USE_MCP"), default=True),
            require_approval=_as_bool(
                os.getenv("IT_OPS_REQUIRE_APPROVAL"), default=True
            ),
            verbose=_as_bool(os.getenv("IT_OPS_VERBOSE"), default=True),
            db_path=_resolve(os.getenv("IT_OPS_DB_PATH", "ops_mcp/it_ops.db")),
            vector_path=_resolve(os.getenv("IT_OPS_VECTOR_PATH", "vector_db")),
            policy_path=_resolve(os.getenv("IT_OPS_POLICY_PATH", "data/policies")),
        )


settings = Settings.from_env()

# reasoning=False turns off the model's thinking mode. The assistant's
# reasoning is the graph itself - plan, gather, decide - so an internal chain of
# thought on top of that only costs latency. num_predict caps a runaway
# generation, which a small model can fall into when a tool result surprises it.
llm = ChatOllama(
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    base_url=settings.ollama_base_url,
    reasoning=False,
    num_predict=settings.max_output_tokens,
)

embeddings = OllamaEmbeddings(
    model=settings.embedding_model,
    base_url=settings.ollama_base_url,
)


def log(message: str) -> None:
    """Print an agent trace line when verbose mode is on."""

    if settings.verbose:
        print(message)
