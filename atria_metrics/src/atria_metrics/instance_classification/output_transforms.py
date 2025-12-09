from atria_core.types import ClassificationModelOutput


def _output_transform(output: ClassificationModelOutput):
    assert isinstance(output, ClassificationModelOutput), (
        f"Expected {ClassificationModelOutput}, got {type(output)}"
    )
    return output.logits, output.gt_label_value
