from atria_hub.api.base import BaseApi
from atria_hub.exceptions import api_error_handler
from atria_hub.utilities import get_logger

logger = get_logger(__name__)


class HealthCheckApi(BaseApi):
    @api_error_handler
    def health_check(self):
        """Perform a health check on the backend."""
        from atriax_client.api.health import health_health_check

        with self._client.api_client as client:
            return health_health_check.sync_detailed(client=client)
