from atria_core.logger import get_logger

logger = get_logger(__name__)


def sign_in(email: str, password: str):
    """Sign in to AtriaX"""
    try:
        from atria_hub.hub import AtriaHub
        from atria_hub.models import AuthLoginModel

        AtriaHub().initialize(AuthLoginModel(email=email, password=password))
    except Exception as e:
        logger.exception(e)
