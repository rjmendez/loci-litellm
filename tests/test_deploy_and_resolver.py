"""Deployment checker output and backend resolution precedence."""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest
import yaml

from loci_litellm import backend_resolver as br
from loci_litellm.backend_resolver import BackendResolver, build_embed_request, parse_embedding

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
_spec = importlib.util.spec_from_file_location("check_deployment_output", DEPLOY / "check_deployment.py")
cd = importlib.util.module_from_spec(_spec)
sys.modules["check_deployment_output"] = cd
_spec.loader.exec_module(cd)


@pytest.fixture
def cfg():
    return yaml.safe_load((DEPLOY / "config.example.yaml").read_text(encoding="utf-8"))


# --- deploy checker ---------------------------------------------------------------------------------

def test_a_clean_run_without_env_says_the_env_half_was_not_checked(tmp_path: Path, cfg, capsys) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert cd.main(["--config", str(path)]) == 0
    out = capsys.readouterr().out
    assert "NOTE" in out and "--env" in out and "not checked" in out
    assert "0 violation(s)" in out


def test_a_run_with_env_does_not_print_the_note(tmp_path: Path, cfg, capsys) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("SOMETHING=1\n", encoding="utf-8")
    assert cd.main(["--config", str(path), "--env", str(env)]) == 0
    assert "NOTE" not in capsys.readouterr().out


def test_ip_behind_an_api_base_variable_is_caught_whatever_the_variable_is_called(cfg) -> None:
    deployment = next(d for d in cfg["model_list"] if d["model_info"]["id"] == "cloud-simple")
    deployment["litellm_params"]["api_base"] = "os.environ/PROVIDER_ENDPOINT_ADDR"
    violations, _ = cd.check(cfg, {"PROVIDER_ENDPOINT_ADDR": "http://203.0.113.9:8080"})
    assert any(v.startswith("no-ip-literal") and "PROVIDER_ENDPOINT_ADDR" in v for v in violations)


def test_a_variable_that_is_neither_named_like_a_url_nor_referenced_is_not_flagged(cfg) -> None:
    violations, _ = cd.check(cfg, {"WORKER_ADDR": "203.0.113.9"})
    assert violations == []


def test_exemption_from_pricing_is_by_variable_name_only(cfg) -> None:
    """The README says so; this pins it so the README cannot silently become untrue."""
    deployment = next(d for d in cfg["model_list"] if d["model_info"]["id"] == "cloud-simple")
    deployment["litellm_params"]["model"] = "openai/not-in-any-cost-map"
    deployment["model_info"].pop("input_cost_per_token", None)
    deployment["model_info"].pop("output_cost_per_token", None)
    deployment["litellm_params"].pop("input_cost_per_token", None)
    deployment["litellm_params"].pop("output_cost_per_token", None)
    deployment["litellm_params"]["api_base"] = "https://api.cloud.example.test/v1"
    violations, _ = cd.check(cfg)
    assert any(v.startswith("priced") for v in violations)
    deployment["litellm_params"]["api_base"] = "os.environ/LOCAL_REMOTE_API"  # name alone grants the exemption
    violations, _ = cd.check(cfg)
    assert not any(v.startswith("priced") and "cloud-simple" in v for v in violations)


def test_checker_makes_no_claim_about_callbacks(cfg) -> None:
    cfg["litellm_settings"]["callbacks"] = []
    violations, warnings = cd.check(cfg)
    assert violations == [] and warnings == []


# --- backend resolver ---------------------------------------------------------------------------------

_EMBED_ENV = ("LOCI_EMBED_API_BASE", "MNEMOSYNE_EMBEDDING_API_URL", "LOCI_EMBED_MODEL", "MNEMOSYNE_EMBEDDING_MODEL",
              "LOCI_EMBED_API_KEY", "EMBED_API_KEY", "LOCI_EMBED_API_PATH", "LOCI_EMBED_AUTH_HEADER",
              "LOCI_EMBED_AUTH_SCHEME", "LOCI_EMBED_EXTRA_HEADERS", "LOCI_EMBED_EXTRA_BODY", "LOCI_EMBED_DIMENSIONS",
              "LOCI_EMBED_QUERY_PREFIX", "LOCI_EMBED_DOCUMENT_PREFIX", "LOCI_QDRANT_URL", "QDRANT_URL",
              "LOCI_QDRANT_API_KEY", "QDRANT_API_KEY")


@pytest.fixture
def clean_env(monkeypatch):
    for name in _EMBED_ENV:
        monkeypatch.delenv(name, raising=False)


def test_embed_api_base_is_used_as_given_minus_a_trailing_slash(monkeypatch, clean_env) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "https://embed.example.test/openai/v1/")
    assert BackendResolver.embed_api_base() == "https://embed.example.test/openai/v1"


def test_loci_embed_base_wins_over_the_mnemosyne_endpoint(monkeypatch, clean_env) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "https://a.example.test/v1")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_API_URL", "https://b.example.test/v1")
    assert BackendResolver.embed_api_base() == "https://a.example.test/v1"
    monkeypatch.delenv("LOCI_EMBED_API_BASE")
    assert BackendResolver.embed_api_base() == "https://b.example.test/v1"


def test_nothing_configured_resolves_to_nothing_not_a_guess(monkeypatch, clean_env) -> None:
    monkeypatch.delenv("LOCI_EMBED_MODEL", raising=False)
    assert BackendResolver.embed_api_base() is None
    assert BackendResolver.embed_model() == ""
    assert BackendResolver.qdrant() == ("", "")
    assert build_embed_request(BackendResolver, "text") is None


def test_a_model_is_required_even_when_the_base_is_set(monkeypatch, clean_env) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "https://embed.example.test/v1")
    monkeypatch.delenv("LOCI_EMBED_MODEL", raising=False)
    assert build_embed_request(BackendResolver, "text") is None


def test_default_request_is_the_openai_shape(monkeypatch, clean_env) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "https://embed.example.test/v1")
    monkeypatch.setenv("LOCI_EMBED_MODEL", "m1")
    monkeypatch.setenv("LOCI_EMBED_API_KEY", "k")
    url, headers, body = build_embed_request(BackendResolver, "hello")
    assert url == "https://embed.example.test/v1/embeddings"
    assert headers == {"Content-Type": "application/json", "Authorization": "Bearer k"}
    assert body == {"model": "m1", "input": "hello"}


def test_every_part_of_the_request_is_configurable(monkeypatch, clean_env) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "https://embed.example.test/deployments/x")
    monkeypatch.setenv("LOCI_EMBED_API_PATH", "/embed?api-version=1")
    monkeypatch.setenv("LOCI_EMBED_MODEL", "m1")
    monkeypatch.setenv("LOCI_EMBED_API_KEY", "k")
    monkeypatch.setenv("LOCI_EMBED_AUTH_HEADER", "api-key")
    monkeypatch.setenv("LOCI_EMBED_AUTH_SCHEME", "")
    monkeypatch.setenv("LOCI_EMBED_EXTRA_HEADERS", '{"x-tenant": "t1"}')
    monkeypatch.setenv("LOCI_EMBED_EXTRA_BODY", '{"input_type": "search_query", "truncate": "END"}')
    monkeypatch.setenv("LOCI_EMBED_DIMENSIONS", "256")
    monkeypatch.setenv("LOCI_EMBED_QUERY_PREFIX", "query: ")
    monkeypatch.setenv("LOCI_EMBED_DOCUMENT_PREFIX", "passage: ")
    url, headers, body = build_embed_request(BackendResolver, "hello", "query")
    assert url == "https://embed.example.test/deployments/x/embed?api-version=1"
    assert headers == {"Content-Type": "application/json", "x-tenant": "t1", "api-key": "k"}
    assert body == {"input_type": "search_query", "truncate": "END", "model": "m1", "input": "query: hello",
                    "dimensions": 256}
    _, _, doc_body = build_embed_request(BackendResolver, "hello", "document")
    assert doc_body["input"] == "passage: hello"


def test_the_embed_section_of_the_config_file_configures_the_same_things(monkeypatch, clean_env, tmp_path: Path) -> None:
    cfg_file = tmp_path / "b.toml"
    cfg_file.write_text(
        '[embed]\napi_base = "https://cfg.example.test/v1"\nmodel = "cfg-model"\napi_key = "cfg-key"\n'
        'dimensions = 64\nquery_prefix = "q: "\n[embed.extra_body]\nencoding_format = "float"\n',
        encoding="utf-8")
    monkeypatch.setattr(br, "_CONFIG_PATH", str(cfg_file))
    monkeypatch.delenv("LOCI_EMBED_MODEL", raising=False)
    br._config.cache_clear()
    url, headers, body = build_embed_request(BackendResolver, "hi")
    assert url == "https://cfg.example.test/v1/embeddings"
    assert headers["Authorization"] == "Bearer cfg-key"
    assert body == {"encoding_format": "float", "model": "cfg-model", "input": "q: hi", "dimensions": 64}
    monkeypatch.setenv("LOCI_EMBED_MODEL", "env-model")
    assert build_embed_request(BackendResolver, "hi")[2]["model"] == "env-model"


def test_malformed_option_values_are_ignored_with_a_warning(monkeypatch, clean_env, caplog) -> None:
    monkeypatch.setenv("LOCI_EMBED_EXTRA_BODY", "{not json")
    monkeypatch.setenv("LOCI_EMBED_EXTRA_HEADERS", "[1, 2]")
    monkeypatch.setenv("LOCI_EMBED_DIMENSIONS", "many")
    with caplog.at_level("WARNING", logger="loci_litellm.backends"):
        opts = BackendResolver.embed_options()
    assert opts["extra_body"] == {} and opts["extra_headers"] == {} and opts["dimensions"] == 0
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 3


@pytest.mark.parametrize("payload,expected", [
    ({"data": [{"embedding": [1.0, 2.0]}]}, [1.0, 2.0]),
    ({"embedding": [3.0]}, [3.0]),
    ({"embeddings": [[4.0, 5.0]]}, [4.0, 5.0]),
    ({"embeddings": {"float": [[6.0]]}}, [6.0]),
])
def test_parse_embedding_accepts_the_common_response_shapes(payload, expected) -> None:
    assert parse_embedding(payload) == expected


@pytest.mark.parametrize("payload", [None, [], {}, {"data": []}, {"embeddings": []}, {"error": "x"}])
def test_parse_embedding_rejects_anything_else(payload) -> None:
    with pytest.raises(ValueError):
        parse_embedding(payload)


def test_loci_prefixed_env_beats_unprefixed_and_both_beat_the_config_file(monkeypatch, clean_env, tmp_path: Path) -> None:
    cfg_file = tmp_path / "b.toml"
    cfg_file.write_text('[qdrant]\nurl = "https://cfg.example.test"\napi_key = "from-config"\n', encoding="utf-8")
    monkeypatch.setattr(br, "_CONFIG_PATH", str(cfg_file))
    br._config.cache_clear()
    assert BackendResolver.qdrant() == ("https://cfg.example.test", "from-config")
    monkeypatch.setenv("QDRANT_URL", "https://plain.example.test")
    assert BackendResolver.qdrant_url() == "https://plain.example.test"
    monkeypatch.setenv("LOCI_QDRANT_URL", "https://loci.example.test")
    assert BackendResolver.qdrant_url() == "https://loci.example.test"


def test_the_conftest_isolates_the_private_config() -> None:
    assert not Path(br._CONFIG_PATH).exists()
    assert br.config_load_error() == ""
