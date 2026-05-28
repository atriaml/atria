def _get_content_type_from_filename(filename: str) -> str:
    import mimetypes

    content_type, _ = mimetypes.guess_type(filename)
    if content_type is None:
        content_type = "application/octet-stream"
    return content_type
