from __future__ import annotations

from typing import TYPE_CHECKING

from atria_hub.api.base import BaseApi

if TYPE_CHECKING:
    import uuid

    from atriax_client.models.task import Task
    from atriax_client.models.task_update import TaskUpdate


class TasksApi(BaseApi):
    def get(self, id: uuid.UUID) -> Task:
        """Retrieve a task from the hub by its name."""

        from atriax_client.api.tasks import tasks_item

        with self._client.protected_api_client as client:
            response = tasks_item.sync_detailed(id, client=client)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get task: {response.status_code} - {response.content.decode('utf-8')}"
                )
            return response.parsed

    def list(self):
        """List all tasks in the hub."""
        from atriax_client.api.tasks import tasks_list

        with self._client.protected_api_client as client:
            response = tasks_list.sync_detailed(client=client)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to list tasks: {response.status_code} - {response.content.decode('utf-8')}"
                )
            return response.parsed

    def update(self, id: uuid.UUID, body: TaskUpdate) -> Task:
        """Update a task in the hub."""

        from atriax_client.api.tasks import tasks_update

        with self._client.protected_api_client as client:
            response = tasks_update.sync_detailed(id=id, body=body, client=client)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to update task: {response.status_code} - {response.content.decode('utf-8')}"
                )
            return response.parsed

    def delete(self, id: uuid.UUID) -> None:
        """Delete a task from the hub."""

        from atriax_client.api.tasks import tasks_delete

        with self._client.protected_api_client as client:
            response = tasks_delete.sync_detailed(id=id, client=client)
            if response.status_code != 204:
                raise RuntimeError(
                    f"Failed to delete task: {response.status_code} - {response.content.decode('utf-8')}"
                )
