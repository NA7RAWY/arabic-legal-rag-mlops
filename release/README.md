# Local canary release

This Compose overlay builds one BentoML application image and runs it as two
independent instances. Nginx is the single client entry point on port 8080 and
routes approximately 90% of requests to `stable` and 10% to `candidate` using
weighted round-robin. Both instances expose the same API and application code;
`DEPLOYMENT_SLOT` only identifies their release role.

```text
client -> nginx:8080 -> stable:3000   (weight 9)
                     -> candidate:3000 (weight 1)
```

By default both services use the locally built `legal-rag-bentoml:local` image.
For a release canary, provide immutable Docker Hub tags without changing the
Compose file:

```bash
export STABLE_IMAGE="<dockerhub-user>/<repository>:v0.2.0"
export CANDIDATE_IMAGE="<dockerhub-user>/<repository>:v0.3.0"
docker compose -f release/docker-compose.canary.yml pull stable candidate
docker compose -f release/docker-compose.canary.yml up -d --no-build
```

The placeholders identify deployment configuration, not credentials. Docker
Hub authentication, when required, should use `docker login` or the deployment
platform's secret store.

## Start and health checks

The local demonstration expects the existing PostgreSQL service to be reachable
through its host-published port. Provider configuration is supplied through the
shell environment; no credentials are stored here.

```bash
docker compose -f release/docker-compose.canary.yml up -d --build

curl http://127.0.0.1:8001/health  # stable diagnostic port
curl http://127.0.0.1:8002/health  # candidate diagnostic port
curl http://127.0.0.1:8080/health  # nginx entry point
```

Open-source nginx provides passive upstream failure handling, not active health
checks. Docker reports each backend's health, while nginx temporarily avoids a
backend after `max_fails=2` proxied failures within `fail_timeout=10s`. Operators
should inspect `docker compose ps`, backend logs, and direct health endpoints.
An unhealthy candidate should trigger the stable-only rollback below rather than
relying on passive behavior as an automated promotion system.

SSE uses a dedicated `/ask/stream` location with proxy buffering and caching
disabled, HTTP/1.1 enabled, and a five-minute upstream read timeout. This allows
chunks to reach clients without nginx accumulating the complete answer.

## Rollback

The repository includes `nginx.stable.conf`, which removes candidate from the
upstream. Switch nginx to it without rebuilding or stopping stable:

```bash
NGINX_CONFIG=./nginx.stable.conf \
  docker compose -f release/docker-compose.canary.yml up -d \
  --force-recreate --no-deps nginx
```

Confirm through `http://127.0.0.1:8080/health`, then stop candidate if desired:

```bash
docker compose -f release/docker-compose.canary.yml stop candidate
```

To restore the reviewed canary configuration, omit `NGINX_CONFIG` and recreate
nginx with the same command.

## Promotion

Promotion is deliberately manual. Edit the two upstream weights in `nginx.conf`,
validate the configuration, and recreate only nginx at each observed stage:

```text
stable 9 / candidate 1  -> 90/10
stable 3 / candidate 1  -> 75/25
stable 1 / candidate 1  -> 50/50
candidate only          -> 0/100
```

Before each step, compare candidate and stable error rate, p95/p99 latency,
streaming error/incomplete-stream rate, container health, and grounded RAG
evaluation signals. No promotion or rollback is automated from fabricated
metrics.

## Strategy distinctions and limitations

- Canary gradually exposes one release candidate to production-like traffic.
- A/B testing intentionally assigns variants to compare user or product outcomes;
  this setup does not provide sticky assignment or experimentation analysis.
- Blue/green keeps two complete environments and switches traffic as one cutover;
  this setup changes weighted traffic progressively.

This is a single-machine demonstration. Nginx, both backends, the shared model
cache, and host PostgreSQL can share the same failure domain and hardware. The
setup does not demonstrate multi-host availability, active nginx health checks,
autoscaling, or production secret management.
