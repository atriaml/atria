from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

from atria_hub.config import settings
from atria_hub.utilities import get_logger

if TYPE_CHECKING:
    from atriax_client import (
        AuthenticatedClient as AuthenticatedAtriaxClient,
        Client as AtriaxClient,
    )
    from lakefs.client import Client as LakeFSClient
    from lakefs_spec import LakeFSFileSystem
    from supabase import Client as SupabaseClient

    from atria_hub.credentials_storage import CredentialsStorage
    from atria_hub.models import ReposCredentials

logger = get_logger(__name__)


class AtriaHubClient:
    def __init__(
        self,
        base_url: str = settings.ATRIAX_URL,
        storage_url: str = settings.ATRIAX_STORAGE_URL,
        service_name: str = "atria",
        use_key_ring: bool = True,
    ):
        from atriax_client import Client as AtriaxClient
        from supabase import Client as AuthClient, ClientOptions, create_client

        from atria_hub.credentials_storage import CredentialsStorage

        self._base_url = base_url
        self._storage_url = storage_url
        self._service_name = service_name
        self._auth_headers: dict[str, str] = {}
        self._credentials_storage = CredentialsStorage(service_name)
        self._api_client = AtriaxClient(base_url=base_url)
        self._auth_client: AuthClient = create_client(
            supabase_url=base_url,
            supabase_key="dummy-key",
            options=ClientOptions(storage=self._credentials_storage)
            if use_key_ring
            else None,
        )
        self._lakefs_client: LakeFSClient | None = None
        self._lakefs_fs: LakeFSFileSystem | None = None

    @property
    def credentials_storage(self) -> CredentialsStorage:
        """Return the credentials storage."""
        return self._credentials_storage

    @property
    def api_client(self) -> AtriaxClient:
        """Return the HTTP client for REST API calls."""
        return self._api_client

    @property
    def protected_api_client(self) -> AuthenticatedAtriaxClient:
        """Return the HTTP client for REST API calls."""
        return self._api_client.with_headers({**self.get_auth_headers()})

    @property
    def auth_client(self) -> SupabaseClient:
        """Return the Supabase client."""
        return self._auth_client

    @cached_property
    def lakefs_client(self) -> LakeFSClient:
        """Return the LakeFS client."""
        from lakefs.client import Client as LakeFSClient

        if self._lakefs_client is None:
            self._lakefs_client = LakeFSClient(host=self._storage_url)
        return self._lakefs_client

    @cached_property
    def fs(self) -> LakeFSFileSystem:
        """Return the LakeFS client."""
        from lakefs_spec import LakeFSFileSystem

        if self._lakefs_fs is None:
            self._lakefs_fs = LakeFSFileSystem(host=self._storage_url)
            self._lakefs_fs.client = self._lakefs_client
        return self._lakefs_fs

    def set_repos_access_credentials(self, credentials: ReposCredentials):
        """Set the credentials in the storage."""
        self.lakefs_client._conf.username = credentials.access_key_id
        self.lakefs_client._conf.password = credentials.secret_access_key

    def get_auth_headers(self) -> dict[str, str]:
        """Return headers using the Supabase session token."""
        session = self._auth_client.auth.get_session()
        if not session:
            raise RuntimeError("No active session. Please authenticate.")
        return {"Authorization": f"Bearer {session.access_token}"}
