---
title: Hub Client
---

# Hub Client

## AtriaHubClient

`AtriaHubClient` is the low-level HTTP client that all hub API objects use. It handles:

- **Base URL configuration** — from `settings.ATRIAX_URL` (env var or default)
- **Storage URL** — separate URL for binary artifact storage (`settings.ATRIAX_STORAGE_URL`)
- **Authentication** — attaches bearer tokens from the keyring to every request
- **Connection checking** — `AtriaHubConnectionError` when the hub is unreachable

## AtriaHub

`AtriaHub` is the high-level facade. It initializes all API objects from a single client:

```python
hub = AtriaHub()
hub.datasets   # DatasetsApi
hub.models     # ModelsApi
hub._auth_api  # AuthApi
```

Most user-facing code interacts with `AtriaHub` via the CLI or via `ModelPipelineOps.push_to_hub()` / `Dataset._hub_ops.push_to_hub()` — direct use of the client is rarely needed.

## API objects

| API | Key operations |
|---|---|
| `DatasetsApi` | `push(dataset, name)`, `pull(name, version)`, `list()` |
| `ModelsApi` | `push(artifact, name)`, `pull(name, version)`, `list()` |
| `AuthApi` | `login(username, password)`, `refresh_token()`, `logout()` |
| `RepoCredentialsApi` | Repository-level access tokens for private artifacts |
| `HealthCheckApi` | `ping()` — checks hub availability |

## Credential storage

After login, credentials are stored in the OS keyring (via the `keyring` library):
- On Linux: `libsecret` or `kwallet`
- On macOS: `Keychain`
- On Windows: `Windows Credential Manager`

The `use_key_ring=True` default means credentials persist across sessions and processes. Pass `use_key_ring=False` to use in-memory credentials only (useful in CI environments).

## Environment configuration

| Env var | Default | Description |
|---|---|---|
| `ATRIAX_URL` | `https://hub.atriaml.com` | Hub API base URL |
| `ATRIAX_STORAGE_URL` | `https://storage.atriaml.com` | Binary artifact storage URL |
