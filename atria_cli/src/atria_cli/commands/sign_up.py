import re

from atria_core.logger import get_logger

logger = get_logger(__name__)


class SignUpError(Exception):
    """Custom exception for sign-up related errors"""

    pass


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def validate_password(password: str) -> tuple[bool, str | None]:
    """
    Validate password strength
    Returns (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one digit"
    return True, None


def sign_up(username: str, email: str, password: str) -> bool:
    """
    Sign up a new user to AtriaX

    Args:
        username: Unique username for the account
        email: Valid email address
        password: Strong password meeting security requirements

    Returns:
        bool: True if sign-up successful, False otherwise

    Raises:
        SignUpError: If validation fails or sign-up process encounters an error
    """
    # Input validation
    if not all([username, email, password]):
        raise SignUpError("All fields are required")

    if not validate_email(email):
        raise SignUpError("Invalid email format")

    is_valid_password, password_error = validate_password(password)
    if not is_valid_password:
        raise SignUpError(f"Password validation failed: {password_error}")

    if len(username.strip()) < 3:
        raise SignUpError("Username must be at least 3 characters long")

    try:
        from atria_hub.hub import AtriaHub

        logger.info(f"Attempting to sign up user: {username}")
        AtriaHub().auth.sign_up(
            email=email.strip().lower(), password=password, username=username.strip()
        )
        logger.info(f"Successfully signed up user: {username}")
        return True

    except Exception as e:
        error_msg = f"Sign-up failed for user {username}: {str(e)}"
        logger.error(error_msg)
        raise SignUpError(error_msg) from e
