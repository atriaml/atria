from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
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
        resource_id (Union[None, UUID, Unset]):
    """

    id: UUID
    created_at: str
    updated_at: str
    type_: UserTaskType
    config: "TaskConfig"
    user_id: UUID
    status: Unset | TaskStatus = UNSET
    error_message: None | Unset | str = UNSET
    resource_id: None | UUID | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        created_at = self.created_at

        updated_at = self.updated_at

        type_ = self.type_.value

        config = self.config.to_dict()

        user_id = str(self.user_id)

        status: Unset | str = UNSET
        if not isinstance(self.status, Unset):
            status = self.status.value

        error_message: None | Unset | str
        if isinstance(self.error_message, Unset):
            error_message = UNSET
        else:
            error_message = self.error_message

        resource_id: None | Unset | str
        if isinstance(self.resource_id, Unset):
            resource_id = UNSET
        elif isinstance(self.resource_id, UUID):
            resource_id = str(self.resource_id)
        else:
            resource_id = self.resource_id

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
        if resource_id is not UNSET:
            field_dict["resource_id"] = resource_id

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
        status: Unset | TaskStatus
        if isinstance(_status, Unset):
            status = UNSET
        else:
            status = TaskStatus(_status)

        def _parse_error_message(data: object) -> None | Unset | str:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | Unset | str, data)

        error_message = _parse_error_message(d.pop("error_message", UNSET))

        def _parse_resource_id(data: object) -> None | UUID | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                resource_id_type_0 = UUID(data)

                return resource_id_type_0
            except:  # noqa: E722
                pass
            return cast(None | UUID | Unset, data)

        resource_id = _parse_resource_id(d.pop("resource_id", UNSET))

        task = cls(
            id=id,
            created_at=created_at,
            updated_at=updated_at,
            type_=type_,
            config=config,
            user_id=user_id,
            status=status,
            error_message=error_message,
            resource_id=resource_id,
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
