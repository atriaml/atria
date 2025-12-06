def _extract_nested_defaults(model: type[BaseModel]) -> dict[str, Any]:
    """
    Recursively extracts default values from a Pydantic BaseModel, including nested models.

    Args:
        model (BaseModel): The Pydantic model instance.
    Returns:
        dict[str, Any]: A dictionary containing field names and their default values.
    """
    defaults = {}
    for field_name, field in model.model_fields.items():
        if field.default is not None:
            if isinstance(field.default, BaseModel):
                defaults[field_name] = _extract_nested_defaults(field.default)
            else:
                defaults[field_name] = field.default
    return defaults
