from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union
from uuid import UUID

from attrs import define as _attrs_define

from ..models.task_type import TaskType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.model_storage_metadata import ModelStorageMetadata


T = TypeVar("T", bound="Model")


@_attrs_define
class Model:
    """
    Attributes:
        id (UUID):
        created_at (str):
        updated_at (str):
        name (str):
        description (str):
        type_ (str):
        repo_id (str):
        user_id (UUID):
        storage_metadata (ModelStorageMetadata):
        task_type (TaskType):
        default_branch (Union[Unset, str]):  Default: 'main'.
        is_public (Union[Unset, bool]):  Default: False.
    """

    id: UUID
    created_at: str
    updated_at: str
    name: str
    description: str
    type_: str
    repo_id: str
    user_id: UUID
    storage_metadata: "ModelStorageMetadata"
    task_type: TaskType
    default_branch: Union[Unset, str] = "main"
    is_public: Union[Unset, bool] = False

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        created_at = self.created_at

        updated_at = self.updated_at

        name = self.name

        description = self.description

        type_ = self.type_

        repo_id = self.repo_id

        user_id = str(self.user_id)

        storage_metadata = self.storage_metadata.to_dict()

        task_type = self.task_type.value

        default_branch = self.default_branch

        is_public = self.is_public

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "id": id,
                "created_at": created_at,
                "updated_at": updated_at,
                "name": name,
                "description": description,
                "type": type_,
                "repo_id": repo_id,
                "user_id": user_id,
                "storage_metadata": storage_metadata,
                "task_type": task_type,
            }
        )
        if default_branch is not UNSET:
            field_dict["default_branch"] = default_branch
        if is_public is not UNSET:
            field_dict["is_public"] = is_public

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.model_storage_metadata import ModelStorageMetadata

        d = dict(src_dict)
        id = UUID(d.pop("id"))

        created_at = d.pop("created_at")

        updated_at = d.pop("updated_at")

        name = d.pop("name")

        description = d.pop("description")

        type_ = d.pop("type")

        repo_id = d.pop("repo_id")

        user_id = UUID(d.pop("user_id"))

        storage_metadata = ModelStorageMetadata.from_dict(d.pop("storage_metadata"))

        task_type = TaskType(d.pop("task_type"))

        default_branch = d.pop("default_branch", UNSET)

        is_public = d.pop("is_public", UNSET)

        model = cls(
            id=id,
            created_at=created_at,
            updated_at=updated_at,
            name=name,
            description=description,
            type_=type_,
            repo_id=repo_id,
            user_id=user_id,
            storage_metadata=storage_metadata,
            task_type=task_type,
            default_branch=default_branch,
            is_public=is_public,
        )

        return model
