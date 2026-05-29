from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.task_status import TaskStatus
from ..models.user_task_type import UserTaskType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.task_config import TaskConfig


T = TypeVar("T", bound="Task")


@_attrs_define
class Task:
    """
    Attributes:
        id (UUID):
        created_at (str):
        updated_at (str):
        type_ (UserTaskType):
        config (TaskConfig):
        user_id (UUID):
        status (Union[Unset, TaskStatus]):
        error_message (Union[None, Unset, str]):
    """

    id: UUID
    created_at: str
    updated_at: str
    type_: UserTaskType
    config: "TaskConfig"
    user_id: UUID
    status: Union[Unset, TaskStatus] = UNSET
    error_message: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        created_at = self.created_at

        updated_at = self.updated_at

        type_ = self.type_.value

        config = self.config.to_dict()

        user_id = str(self.user_id)

        status: Union[Unset, str] = UNSET
        if not isinstance(self.status, Unset):
            status = self.status.value

        error_message: Union[None, Unset, str]
        if isinstance(self.error_message, Unset):
            error_message = UNSET
        else:
            error_message = self.error_message

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "created_at": created_at,
                "updated_at": updated_at,
                "type": type_,
                "config": config,
                "user_id": user_id,
            }
        )
        if status is not UNSET:
            field_dict["status"] = status
        if error_message is not UNSET:
            field_dict["error_message"] = error_message

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.task_config import TaskConfig

        d = dict(src_dict)
        id = UUID(d.pop("id"))

        created_at = d.pop("created_at")

        updated_at = d.pop("updated_at")

        type_ = UserTaskType(d.pop("type"))

        config = TaskConfig.from_dict(d.pop("config"))

        user_id = UUID(d.pop("user_id"))

        _status = d.pop("status", UNSET)
        status: Union[Unset, TaskStatus]
        if isinstance(_status, Unset):
            status = UNSET
        else:
            status = TaskStatus(_status)

        def _parse_error_message(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        error_message = _parse_error_message(d.pop("error_message", UNSET))

        task = cls(
            id=id,
            created_at=created_at,
            updated_at=updated_at,
            type_=type_,
            config=config,
            user_id=user_id,
            status=status,
            error_message=error_message,
        )

        task.additional_properties = d
        return task

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
