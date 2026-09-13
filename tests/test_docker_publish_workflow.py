"""Static checks for Docker Hub publishing safeguards and traceability."""

from pathlib import Path

WORKFLOW = Path(".github/workflows/ci.yml")


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_publish_runs_after_existing_ci_gates() -> None:
    workflow = _workflow_text()

    assert "pull_request:" in workflow
    assert "branches: [main]" in workflow
    assert '"v[0-9]+.[0-9]+.[0-9]+"' in workflow
    assert "needs: [quality, docker-build]" in workflow
    assert "if: github.event_name == 'push'" in workflow


def test_publish_uses_docker_hub_secrets_and_repository_variable() -> None:
    workflow = _workflow_text()

    assert "secrets.DOCKERHUB_USERNAME" in workflow
    assert "secrets.DOCKERHUB_TOKEN" in workflow
    assert "vars.DOCKERHUB_REPOSITORY" in workflow
    assert "docker/login-action@v3" in workflow


def test_publish_builds_canary_image_with_traceable_tags_and_labels() -> None:
    workflow = _workflow_text()

    assert "docker/setup-buildx-action@v3" in workflow
    assert "docker/metadata-action@v5" in workflow
    assert "docker/build-push-action@v6" in workflow
    assert "file: release/Dockerfile" in workflow
    assert "platforms: linux/amd64" in workflow
    assert "type=semver,pattern={{version}}" in workflow
    assert "type=sha,prefix=sha-,format=short" in workflow
    assert "org.opencontainers.image.revision" in workflow
    assert "org.opencontainers.image.source" in workflow
    assert "org.opencontainers.image.version" in workflow
