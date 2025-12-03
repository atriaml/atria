from importlib import import_module


def _resolve_module_from_path(module_path: str) -> type:
    """
    Resolves a class or function from a module path string.

    Args:
        module_path (str): The module path in the format 'module_name.class_name'.

    Returns:
        object: The resolved class or function.

    Raises:
        ValueError: If the module path is invalid or cannot be resolved.
    """
    path = module_path.rsplit(".", 1)
    if len(path) == 1:
        raise ValueError(
            f"Invalid module path: {module_path}. It should be in the form 'module_name.class_name'."
        )
    module_name, class_name = path
    module = import_module(module_name)
    return getattr(module, class_name)
