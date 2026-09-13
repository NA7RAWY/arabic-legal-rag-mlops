"""Static validation for the local nginx canary release configuration."""

from pathlib import Path

RELEASE_DIR = Path("release")


def test_canary_release_files_exist() -> None:
    for relative_path in (
        "Dockerfile",
        "nginx.conf",
        "nginx.stable.conf",
        "docker-compose.canary.yml",
        "README.md",
    ):
        assert (RELEASE_DIR / relative_path).is_file()


def test_nginx_uses_weighted_stable_candidate_upstream() -> None:
    config = (RELEASE_DIR / "nginx.conf").read_text(encoding="utf-8")

    assert "server stable:3000 weight=9" in config
    assert "server candidate:3000 weight=1" in config
    assert "proxy_pass http://legal_rag_backends;" in config


def test_nginx_streaming_location_disables_buffering_and_cache() -> None:
    config = (RELEASE_DIR / "nginx.conf").read_text(encoding="utf-8")
    streaming = config.split("location = /ask/stream", maxsplit=1)[1].split(
        "location /", maxsplit=1
    )[0]

    assert "proxy_http_version 1.1;" in streaming
    assert "proxy_buffering off;" in streaming
    assert "proxy_cache off;" in streaming
    assert "proxy_read_timeout 300s;" in streaming


def test_rollback_configuration_routes_only_to_stable() -> None:
    config = (RELEASE_DIR / "nginx.stable.conf").read_text(encoding="utf-8")

    assert "server stable:3000;" in config
    assert "server candidate:" not in config


def test_compose_runs_same_image_as_distinct_release_slots() -> None:
    compose = (RELEASE_DIR / "docker-compose.canary.yml").read_text(encoding="utf-8")

    assert "stable:" in compose
    assert "candidate:" in compose
    assert "image: ${STABLE_IMAGE:-legal-rag-bentoml:local}" in compose
    assert "image: ${CANDIDATE_IMAGE:-legal-rag-bentoml:local}" in compose
    assert "DEPLOYMENT_SLOT: stable" in compose
    assert "DEPLOYMENT_SLOT: candidate" in compose
    assert "${NGINX_CONFIG:-./nginx.conf}" in compose
