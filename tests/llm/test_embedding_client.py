import base64
import struct
from types import SimpleNamespace
from typing import Any

import pytest

from src.config import EmbeddingModelConfig, resolve_embedding_model_config
from src.embedding_client import _EmbeddingClient  # pyright: ignore[reportPrivateUsage]


class FakeOpenAIEmbeddingsAPI:
    def __init__(self, embedding: list[float] | str) -> None:
        self.embedding: list[float] | str = embedding
        self.calls: list[dict[str, Any]] = []

    async def create(
        self,
        *,
        model: str,
        input: str | list[str],
        **kwargs: Any,
    ) -> SimpleNamespace:
        call: dict[str, Any] = {"model": model, "input": input}
        call.update(kwargs)
        self.calls.append(call)
        if isinstance(input, list):
            data = [SimpleNamespace(embedding=self.embedding) for _ in input]
        else:
            data = [SimpleNamespace(embedding=self.embedding)]
        return SimpleNamespace(data=data)


@pytest.mark.asyncio
async def test_openai_embedding_client_uses_configured_model_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 8)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
            base_url="http://localhost:8000/v1",
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    embedding = await client.embed("hello world")

    assert embedding == [0.1] * 8
    assert fake_embeddings.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["hello world"],
            "encoding_format": "base64",
        }
    ]


@pytest.mark.asyncio
async def test_openai_embedding_client_rejects_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_embeddings = FakeOpenAIEmbeddingsAPI([0.1] * 7)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        ),
        vector_dimensions=8,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        await client.embed("hello world")


@pytest.mark.asyncio
async def test_gemini_embedding_client_uses_output_dimensionality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeGeminiModels:
        async def embed_content(
            self,
            *,
            model: str,
            contents: str | list[str],
            config: dict[str, Any],
        ) -> SimpleNamespace:
            calls.append(
                {
                    "model": model,
                    "contents": contents,
                    "config": config,
                }
            )
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[0.2] * 12)],
            )

    class FakeGeminiClient:
        def __init__(self, *, api_key: str | None, http_options: Any) -> None:
            self.api_key: str | None = api_key
            self.http_options: Any = http_options
            self.aio: Any = SimpleNamespace(models=FakeGeminiModels())

    monkeypatch.setattr("src.embedding_client.genai.Client", FakeGeminiClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="gemini",
            model="gemini-embedding-001",
            api_key="gemini-key",
            base_url="https://gemini-proxy.example/v1beta",
        ),
        vector_dimensions=12,
        max_input_tokens=4096,
        max_tokens_per_request=300_000,
        send_dimensions=False,
    )

    embedding = await client.embed("hello world")

    assert embedding == [0.2] * 12
    assert calls == [
        {
            "model": "gemini-embedding-001",
            "contents": "hello world",
            "config": {"output_dimensionality": 12},
        }
    ]


def _build_openai_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embedding: list[float] | str,
    model: str,
    send_dimensions: bool,
    vector_dimensions: int,
    encoding_format: str | None = "base64",
) -> tuple[_EmbeddingClient, FakeOpenAIEmbeddingsAPI]:
    fake_embeddings = FakeOpenAIEmbeddingsAPI(embedding)

    class FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.embeddings: FakeOpenAIEmbeddingsAPI = fake_embeddings

    monkeypatch.setattr("src.embedding_client.AsyncOpenAI", FakeOpenAIClient)

    client = _EmbeddingClient(
        EmbeddingModelConfig(
            transport="openai",
            model=model,
            api_key="test-key",
            encoding_format=encoding_format,  # pyright: ignore[reportArgumentType]
        ),
        vector_dimensions=vector_dimensions,
        max_input_tokens=8192,
        max_tokens_per_request=300_000,
        send_dimensions=send_dimensions,
    )
    return client, fake_embeddings


@pytest.mark.asyncio
async def test_openai_embed_forwards_dimensions_when_send_dimensions_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.embed("hello")

    assert fake.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["hello"],
            "dimensions": 768,
            "encoding_format": "base64",
        }
    ]


@pytest.mark.asyncio
async def test_openai_embed_omits_dimensions_when_send_dimensions_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 1536,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=1536,
    )

    await client.embed("hello")

    assert fake.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["hello"],
            "encoding_format": "base64",
        }
    ]


@pytest.mark.asyncio
async def test_openai_simple_batch_embed_forwards_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.simple_batch_embed(["a", "b"])

    assert len(fake.calls) == 1
    assert fake.calls[0]["dimensions"] == 768
    assert fake.calls[0]["encoding_format"] == "base64"
    assert fake.calls[0]["input"] == ["a", "b"]


@pytest.mark.asyncio
async def test_openai_batch_embed_forwards_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 768,
        model="text-embedding-3-small",
        send_dimensions=True,
        vector_dimensions=768,
    )

    await client.batch_embed({"a": "hello", "b": "world"})

    assert len(fake.calls) == 1
    assert fake.calls[0]["dimensions"] == 768
    assert fake.calls[0]["encoding_format"] == "base64"


@pytest.mark.asyncio
async def test_openai_embed_forwards_float_encoding_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 1536,
        model="google/gemini-embedding-2-preview",
        send_dimensions=True,
        vector_dimensions=1536,
        encoding_format="float",
    )

    embedding = await client.embed("hello")

    assert len(embedding) == 1536
    assert fake.calls == [
        {
            "model": "google/gemini-embedding-2-preview",
            "input": ["hello"],
            "dimensions": 1536,
            "encoding_format": "float",
        }
    ]


@pytest.mark.asyncio
async def test_openai_embed_omits_empty_encoding_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, fake = _build_openai_client(
        monkeypatch,
        embedding=[0.1] * 1536,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=1536,
        encoding_format=None,
    )

    await client.embed("hello")

    assert fake.calls == [{"model": "text-embedding-3-small", "input": ["hello"]}]


@pytest.mark.asyncio
async def test_openai_embed_accepts_base64_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = base64.b64encode(struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)).decode("ascii")
    client, _fake = _build_openai_client(
        monkeypatch,
        embedding=encoded,
        model="text-embedding-3-small",
        send_dimensions=False,
        vector_dimensions=4,
    )

    embedding = await client.embed("hello")

    assert embedding == pytest.approx([0.1, 0.2, 0.3, 0.4])


def _build_embedding_settings(
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Construct a fresh EmbeddingSettings from the given env, isolated from os.environ."""
    from src.config import EmbeddingSettings

    for key in (
        "EMBEDDING_VECTOR_DIMENSIONS",
        "EMBEDDING_MODEL_CONFIG__MODEL",
        "EMBEDDING_MODEL_CONFIG__TRANSPORT",
        "EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE",
        "EMBEDDING_MODEL_CONFIG__ENCODING_FORMAT",
        "EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL",
        "EMBEDDING_ENCODING_FORMAT",
        "EMBEDDING_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return EmbeddingSettings()


def test_resolve_send_dimensions_auto_default_dim_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings({}, monkeypatch)
    assert s.resolve_send_dimensions() is False


def test_resolve_send_dimensions_auto_explicit_dim_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings({"EMBEDDING_VECTOR_DIMENSIONS": "768"}, monkeypatch)
    assert s.resolve_send_dimensions() is True


def test_resolve_send_dimensions_auto_ada_002_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {
            "EMBEDDING_VECTOR_DIMENSIONS": "1536",
            "EMBEDDING_MODEL_CONFIG__MODEL": "text-embedding-ada-002",
        },
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is False


def test_resolve_send_dimensions_always_returns_true_regardless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {"EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE": "always"},
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is True


def test_resolve_send_dimensions_always_overrides_ada_rejecting_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {
            "EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE": "always",
            "EMBEDDING_MODEL_CONFIG__MODEL": "text-embedding-ada-002",
        },
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is True


def test_resolve_send_dimensions_never_returns_false_regardless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {
            "EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE": "never",
            "EMBEDDING_VECTOR_DIMENSIONS": "768",
        },
        monkeypatch,
    )
    assert s.resolve_send_dimensions() is False


def test_embedding_encoding_format_defaults_to_base64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings({}, monkeypatch)
    assert s.MODEL_CONFIG.encoding_format == "base64"


def test_embedding_encoding_format_env_override_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {"EMBEDDING_MODEL_CONFIG__ENCODING_FORMAT": "float"},
        monkeypatch,
    )
    assert s.MODEL_CONFIG.encoding_format == "float"


def test_embedding_encoding_format_empty_env_omits_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {"EMBEDDING_MODEL_CONFIG__ENCODING_FORMAT": ""},
        monkeypatch,
    )
    assert s.MODEL_CONFIG.encoding_format is None


def test_embedding_top_level_encoding_format_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings({"EMBEDDING_ENCODING_FORMAT": "float"}, monkeypatch)
    assert s.MODEL_CONFIG.encoding_format == "float"


def test_embedding_top_level_base_url_env_populates_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _build_embedding_settings(
        {"EMBEDDING_BASE_URL": "https://openrouter.ai/api/v1"},
        monkeypatch,
    )
    resolved = resolve_embedding_model_config(s.MODEL_CONFIG)
    assert resolved.base_url == "https://openrouter.ai/api/v1"
