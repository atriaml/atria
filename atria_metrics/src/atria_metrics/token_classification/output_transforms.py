from atria_core.types import TokenClassificationModelOutput


def _output_transform(model_output: TokenClassificationModelOutput):
    assert isinstance(model_output, TokenClassificationModelOutput), (
        f"Expected {TokenClassificationModelOutput}, got {type(model_output)}"
    )
    return model_output.target_label_names, model_output.predicted_label_names
