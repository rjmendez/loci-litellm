"""The deployment invariants must fail loudly when broken (see deploy/check_deployment.py)."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"
spec = importlib.util.spec_from_file_location("check_deployment", DEPLOY / "check_deployment.py")
cd = importlib.util.module_from_spec(spec)
sys.modules["check_deployment"] = cd
spec.loader.exec_module(cd)


@pytest.fixture
def cfg():
    return yaml.safe_load((DEPLOY / "config.example.yaml").read_text(encoding="utf-8"))


def rules(violations):
    return {v.split(":")[0] for v in violations}


def test_example_config_is_clean(cfg):
    violations, warnings = cd.check(cfg, {"LOCAL_LLM_API_BASE": "https://ollama.example.test:4443"})
    assert violations == [] and warnings == []


def test_missing_embedding_group(cfg):
    cfg["model_list"] = [d for d in cfg["model_list"] if d["model_name"] != "embedding"]
    cfg["router_settings"]["model_group_alias"] = {}
    v, _ = cd.check(cfg)
    assert rules(v) == {"embedding-group"}


def test_embedding_group_in_fallback_chain(cfg):
    cfg["router_settings"]["fallbacks"].append({"embedding": ["medium-tier"]})
    v, _ = cd.check(cfg)
    assert "embedding-group" in rules(v)


def test_embedding_group_as_router_tier(cfg):
    auto = next(d for d in cfg["model_list"] if d["model_name"] == "auto-router")
    auto["litellm_params"]["complexity_router_config"]["tiers"]["SIMPLE"] = "embedding"
    v, _ = cd.check(cfg)
    assert "embedding-group" in rules(v)


def test_unpriced_cloud_deployment(cfg, monkeypatch):
    monkeypatch.setattr(cd, "_cost_map_prices", lambda model: False)
    dep = next(d for d in cfg["model_list"] if d["model_info"]["id"] == "cloud-medium")
    del dep["model_info"]["output_cost_per_token"]
    v, _ = cd.check(cfg)
    assert rules(v) == {"priced"} and "cloud-medium" in v[0]


def test_cost_map_counts_as_priced(cfg, monkeypatch):
    dep = next(d for d in cfg["model_list"] if d["model_info"]["id"] == "cloud-medium")
    del dep["model_info"]["input_cost_per_token"], dep["model_info"]["output_cost_per_token"]
    monkeypatch.setattr(cd, "_cost_map_prices", lambda model: model == dep["litellm_params"]["model"])
    v, _ = cd.check(cfg)
    assert v == []


def test_local_and_embedding_deployments_need_no_price(cfg):
    v, _ = cd.check(cfg)
    assert not any("local-simple" in x or "embed-default" in x for x in v)


@pytest.mark.parametrize("base", ["http://198.51.100.7:4443/v1", "http://127.0.0.1:11435", "https://[::1]:4443"])
def test_ip_literal_api_base_is_rejected(cfg, base):
    next(d for d in cfg["model_list"] if d["model_info"]["id"] == "cloud-simple")["litellm_params"]["api_base"] = base
    v, _ = cd.check(cfg)
    assert "no-ip-literal" in rules(v)


@pytest.mark.parametrize("key,value", [
    ("LOCAL_LLM_API_BASE", "http://127.0.0.1:11435"),
    ("LOCI_EMBED_API_BASE", "http://198.51.100.7:11434"),
    ("LOCAL_EMBED_API_BASE", "http://203.0.113.9:11434"),
])
def test_ip_literal_in_env_is_rejected(cfg, key, value):
    v, _ = cd.check(cfg, {key: value})
    assert rules(v) == {"no-ip-literal"} and key in v[0]


def test_names_and_bind_variables_pass(cfg):
    env = {"LOCAL_LLM_API_BASE": "https://ollama.lab.example.test:4443",
           "LITELLM_BIND_HOST": "127.0.0.1",          # a listener bind, not a consumer URL
           "LITELLM_BIND_PORT": "4000"}
    v, _ = cd.check(cfg, env)
    assert v == []


def test_alias_to_missing_group(cfg):
    cfg["router_settings"]["model_group_alias"]["ghost"] = "no-such-group"
    v, _ = cd.check(cfg)
    assert rules(v) == {"alias-targets"}


def test_missing_trusted_proxy_ranges_is_a_warning_not_a_failure(cfg):
    del cfg["general_settings"]["trusted_proxy_ranges"]
    v, w = cd.check(cfg)
    assert v == [] and rules(w) == {"trusted-proxies"}


def test_cli_exit_codes(tmp_path, cfg, capsys):
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert cd.main(["--config", str(good)]) == 0
    bad = copy.deepcopy(cfg)
    bad["model_list"] = [d for d in bad["model_list"] if d["model_name"] != "embedding"]
    bad["router_settings"]["model_group_alias"] = {}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    assert cd.main(["--config", str(path)]) == 1
    assert "embedding-group" in capsys.readouterr().out
