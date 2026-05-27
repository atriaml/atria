from atria_core.logger import get_logger

logger = get_logger(__name__)


def sign_out():
    """Sign out from atriax"""
    try:
        from atria_hub.hub import AtriaHub

        AtriaHub().auth.sign_out()
    except Exception as e:
        logger.exception(e)
