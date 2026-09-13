# Docker image publishing

GitHub Actions validates the regular FastAPI production image on every pull
request and every push to `main`. Pull requests never authenticate to Docker Hub
and never push images.

After the quality and Docker build jobs pass, pushes to `main` publish the
BentoML serving image for the canary stack with these tags:

- `main`
- `sha-<short-commit-sha>`

A semantic release tag such as `v0.3.0` publishes:

- `v0.3.0`
- `0.3.0`
- `sha-<short-commit-sha>`
- `latest`

`latest` is therefore reserved for an explicit version release and is not
updated by pull requests or ordinary `main` builds. Images include OCI source,
revision, and version labels so they remain traceable to Git.

## GitHub configuration

Configure these repository settings before enabling publishing:

- Secret `DOCKERHUB_USERNAME`: Docker Hub account or organization username.
- Secret `DOCKERHUB_TOKEN`: scoped Docker Hub access token with push access.
- Variable `DOCKERHUB_REPOSITORY`: repository name, for example
  `arabic-legal-rag`.

Credentials must not be stored in Git, workflow files, `.env`, or image layers.
The milestone intentionally targets Docker Hub only and builds `linux/amd64`.
Registry promotion, signing, and multi-architecture publishing can be added when
deployment requirements justify them.

## Local build and canary use

Build the same BentoML image locally without publishing it:

```bash
docker build -f release/Dockerfile -t arabic-legal-rag:module3-test .
```

The canary Compose configuration accepts `STABLE_IMAGE` and `CANDIDATE_IMAGE`.
Set them to immutable published tags so nginx can route between known releases:

```bash
export STABLE_IMAGE="<dockerhub-user>/<repository>:v0.2.0"
export CANDIDATE_IMAGE="<dockerhub-user>/<repository>:v0.3.0"
docker compose -f release/docker-compose.canary.yml pull stable candidate
docker compose -f release/docker-compose.canary.yml up -d --no-build
```

See `release/README.md` for weighted rollout and rollback procedures.
