from gotrue.types import User

from atria_hub.api.base import BaseApi
from atria_hub.utilities import get_logger

logger = get_logger(__name__)


class AuthApi(BaseApi):
    @property
    def user(self) -> User:
        """Return the user ID from the Supabase session."""
        session = self.get_session()
        if not session:
            raise RuntimeError("No active session. Please authenticate.")
        return session.user

    @property
    def username(self) -> User:
        """Return the user ID from the Supabase session."""
        session = self.get_session()
        if not session:
            raise RuntimeError("No active session. Please authenticate.")
        return session.user.user_metadata["profile"]["username"]

    def get_session(self):
        try:
            return self._client.auth_client.auth.get_session()
        except Exception:
            return None

    def initialize_auth(
        self,
        email: str | None = None,
        password: str | None = None,
        force_sign_in: bool = False,
    ):
        """Authenticate user and store the session."""
        session = self.get_session()
        if session and email is not None and password is not None:
            if session.user.email == email:
                logger.info(f"Already signed in as {email}")
                return
            else:
                force_sign_in = True

        if not session or force_sign_in:
            if email is None or password is None:
                email = input("Enter your email: ")
                password = input("Enter your password: ")

            self.sign_in(email=email, password=password)

    def sign_in(self, email: str, password: str) -> User:
        """Sign in the user."""
        result = self._client.auth_client.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        if not result.session or not result.user:
            raise RuntimeError("Sign-in failed. Please check your credentials.")
        logger.info(f"Sign-in successful for {email}")
        return result.user

    def sign_up(
        self, email: str, password: str, username: str, full_name: str | None = None
    ):
        """Sign up a new user."""
        result = self._client.auth_client.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": {"data": {"username": username, "full_name": full_name}},
            }
        )
        if not result.session or not result.user:
            raise RuntimeError("Sign-up failed. Please check your credentials.")
        logger.info(f"Sign-up successful for {email}")
        return result.user

    def sign_out(self):
        """Sign out the user by removing the session."""
        session = self.get_session()
        if not session:
            logger.warning("No active session to sign out from.")
            return
        self._client.auth_client.auth.sign_out()
        logger.info(
            "Successfully signed out for user: %s",
            session.user.email if session else "Unknown",
        )
