from atria_hub.api.base import BaseApi
from atria_hub.models import ReposCredentials
from atria_hub.utilities import get_logger

logger = get_logger(__name__)


class RepoCredentialsApi(BaseApi):
    def get_or_create(self) -> ReposCredentials:
        """Initialize storage API access."""
        try:
            credentials = self._get_stored_credentials()
            if credentials is None or not self._validate_credentials(
                credentials.access_key_id
            ):
                credentials = self._create_and_store_credentials()
            return credentials
        except Exception as e:
            logger.error("Failed to get or create credentials")
            raise e

    def _get_stored_credentials(self) -> ReposCredentials | None:
        """Get stored credentials from keyring."""
        access_key_id = self._client.credentials_storage.get_item("access_key_id")
        secret_access_key = self._client.credentials_storage.get_item(
            "secret_access_key"
        )

        if access_key_id and secret_access_key:
            return ReposCredentials(
                access_key_id=access_key_id, secret_access_key=secret_access_key
            )
        return None

    def _validate_credentials(self, access_key_id: str) -> bool:
        """Validate if credentials are still valid."""
        try:
            from atriax_client.api.credentials import credentials_get

            with self._client.protected_api_client as client:
                response = credentials_get.sync_detailed(access_key_id, client=client)
                if response.status_code == 404:
                    return False
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to validate credentials: {response.status_code} - {response.content.decode('utf-8')}"
                    )
                return True
        except Exception as e:
            logger.info(f"Credentials validation failed: {e}")
            return False

    def _create_and_store_credentials(self):
        """Create new credentials and store them in keyring."""
        from atriax_client.api.credentials import credentials_create_credentials

        with self._client.protected_api_client as client:
            response = credentials_create_credentials.sync_detailed(client=client)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to validate credentials: {response.status_code} - {response.content.decode('utf-8')}"
                )

            credentials = response.parsed
            self._client.credentials_storage.set_item(
                "access_key_id", response.parsed.access_key_id
            )
            self._client.credentials_storage.set_item(
                "secret_access_key", response.parsed.secret_access_key
            )

            return ReposCredentials(
                access_key_id=credentials.access_key_id,
                secret_access_key=credentials.secret_access_key,
            )
