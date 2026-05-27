from __future__ import annotations

from typing import TYPE_CHECKING

from atria_hub.api.base import BaseApi

if TYPE_CHECKING:
    import uuid

    from atriax_client.models.config import Config


class ConfigSnapshotsApi(BaseApi):
    def get(self, id: uuid.UUID) -> Config:
        """Retrieve a config_snapshot from the hub by its name."""

        from atriax_client.api.config_snapshots import config_snapshots_item

        with self._client.protected_api_client as client:
            response = config_snapshots_item.sync_detailed(id, client=client)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get config_snapshot: {response.status_code} - {response.content.decode('utf-8')}"
                )
            print("response", response, response.parsed, flush=True)
            return response.parsed

    def delete(self, id: uuid.UUID) -> None:
        """Delete a config_snapshot from the hub."""

        from atriax_client.api.config_snapshots import config_snapshots_delete

        with self._client.protected_api_client as client:
            response = config_snapshots_delete.sync_detailed(id=id, client=client)
            if response.status_code != 204:
                raise RuntimeError(
                    f"Failed to delete config_snapshot: {response.status_code} - {response.content.decode('utf-8')}"
                )
