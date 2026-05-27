from pydantic import BaseModel, EmailStr


class AuthLoginModel(BaseModel):
    email: EmailStr
    password: str


class ReposCredentials(BaseModel):
    access_key_id: str
    secret_access_key: str
