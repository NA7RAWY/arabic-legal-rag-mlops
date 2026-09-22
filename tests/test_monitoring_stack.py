"""Static validation for local Prometheus and Grafana integration."""

import json
from pathlib import Path

COMPOSE_PATH = Path("compose.yaml")
PROMETHEUS_CONFIG = Path("monitoring/prometheus/prometheus.yml")
DATASOURCE_CONFIG = Path("monitoring/grafana/provisioning/datasources/prometheus.yml")
DASHBOARD_PROVIDER = Path("monitoring/grafana/provisioning/dashboards/dashboards.yml")
DASHBOARD_PATH = Path("monitoring/grafana/dashboards/legal-rag-observability.json")
DOCKERFILE_PATH = Path("Dockerfile")


def test_monitoring_configuration_files_exist() -> None:
    for path in (
        PROMETHEUS_CONFIG,
        DATASOURCE_CONFIG,
        DASHBOARD_PROVIDER,
        DASHBOARD_PATH,
    ):
        assert path.is_file()


def test_baseline_image_installs_session4_runtime_dependencies() -> None:
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert '"prometheus-client>=0.21,<1"' in dockerfile
    assert '"langfuse>=4,<5"' in dockerfile
    assert '"anyio>=4.10,<5"' in dockerfile


def test_prometheus_scrapes_host_application_without_container_localhost() -> None:
    config = PROMETHEUS_CONFIG.read_text(encoding="utf-8")
    compose = COMPOSE_PATH.read_text(encoding="utf-8")

    assert "job_name: legal-rag-api" in config
    assert "metrics_path: /metrics" in config
    assert "host.docker.internal:8000" in config
    assert "localhost:8000" not in config
    assert '"host.docker.internal:host-gateway"' in compose


def test_compose_provisions_lightweight_persistent_monitoring_services() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")

    assert "prom/prometheus:v3.5.5" in compose
    assert "grafana/grafana:12.4.11" in compose
    assert "prometheus_data:/prometheus" in compose
    assert "grafana_data:/var/lib/grafana" in compose
    assert "storage.tsdb.retention.time=7d" in compose
    assert "http://localhost:9090/-/healthy" in compose
    assert "http://localhost:3000/api/health" in compose
    assert "alertmanager" not in compose.lower()


def test_grafana_datasource_and_dashboard_are_file_provisioned() -> None:
    datasource = DATASOURCE_CONFIG.read_text(encoding="utf-8")
    provider = DASHBOARD_PROVIDER.read_text(encoding="utf-8")

    assert "uid: legal-rag-prometheus" in datasource
    assert "url: http://prometheus:9090" in datasource
    assert "isDefault: true" in datasource
    assert "path: /var/lib/grafana/dashboards" in provider
    assert "allowUiUpdates: false" in provider


def test_dashboard_contains_expected_legal_rag_panels_and_queries() -> None:
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    panels = dashboard["panels"]
    titles = {panel["title"] for panel in panels}
    expressions = "\n".join(
        target["expr"] for panel in panels for target in panel["targets"]
    )

    assert titles == {
        "HTTP request rate",
        "HTTP error rate",
        "p95 HTTP latency",
        "/ask request rate by mode",
        "p95 retrieval latency",
        "p95 generation latency",
        "Mean retrieved source count",
        "LLM provider failure rate",
        "Application process CPU",
        "Application process memory",
        "Authoritative token usage rate",
        "Configured LLM cost per hour",
    }
    for metric in (
        "legal_rag_http_requests_total",
        "legal_rag_http_errors_total",
        "legal_rag_http_request_duration_seconds_bucket",
        "legal_rag_ask_requests_total",
        "legal_rag_retrieval_duration_seconds_bucket",
        "legal_rag_generation_duration_seconds_bucket",
        "legal_rag_retrieved_sources_sum",
        "legal_rag_llm_provider_failures_total",
        "process_cpu_seconds_total",
        "process_resident_memory_bytes",
        "legal_rag_llm_tokens_total",
        "legal_rag_llm_usage_cost_usd_total",
    ):
        assert metric in expressions
    assert 'token_type=~"input|output"' in expressions
    assert "* 3600" in expressions


def test_token_panel_is_explicit_about_authoritative_usage_requirement() -> None:
    dashboard = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    token_panel = next(
        panel
        for panel in dashboard["panels"]
        if panel["title"] == "Authoritative token usage rate"
    )

    assert "authoritative" in token_panel["description"]

    cost_panel = next(
        panel
        for panel in dashboard["panels"]
        if panel["title"] == "Configured LLM cost per hour"
    )
    assert "operator-configured" in cost_panel["description"]
    assert "does not mean free" in cost_panel["description"]
