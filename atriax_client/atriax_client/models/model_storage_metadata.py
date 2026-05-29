from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.lake_fs_branch_summary import LakeFSBranchSummary
    from ..models.model_storage_metadata_configs import ModelStorageMetadataConfigs


T = TypeVar("T", bound="ModelStorageMetadata")


@_attrs_define
class ModelStorageMetadata:
    """
    Attributes:
        main_branch (str):
        branches (list['LakeFSBranchSummary']):
        configs (Union[Unset, ModelStorageMetadataConfigs]):
    """

    main_branch: str
    branches: list["LakeFSBranchSummary"]
    configs: Union[Unset, "ModelStorageMetadataConfigs"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        main_branch = self.main_branch

        branches = []
        for branches_item_data in self.branches:
            branches_item = branches_item_data.to_dict()
            branches.append(branches_item)

        configs: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.configs, Unset):
            configs = self.configs.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "main_branch": main_branch,
                "branches": branches,
            }
        )
        if configs is not UNSET:
            field_dict["configs"] = configs

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.lake_fs_branch_summary import LakeFSBranchSummary
        from ..models.model_storage_metadata_configs import ModelStorageMetadataConfigs

        d = dict(src_dict)
        main_branch = d.pop("main_branch")

        branches = []
        _branches = d.pop("branches")
        for branches_item_data in _branches:
            branches_item = LakeFSBranchSummary.from_dict(branches_item_data)

            branches.append(branches_item)

        _configs = d.pop("configs", UNSET)
        configs: Union[Unset, ModelStorageMetadataConfigs]
        if isinstance(_configs, Unset):
            configs = UNSET
        else:
            configs = ModelStorageMetadataConfigs.from_dict(_configs)

        model_storage_metadata = cls(
            main_branch=main_branch,
            branches=branches,
            configs=configs,
        )

        model_storage_metadata.additional_properties = d
        return model_storage_metadata

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
