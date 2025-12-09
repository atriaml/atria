from atria_core.types import MMDetEvaluationOutput

from atria_metrics.detection.typing import GroundTruthInstances, PredInstances


def _cocoeval_output_transform(model_output: MMDetEvaluationOutput):
    from mmdet.structures.bbox import scale_boxes

    assert isinstance(model_output, MMDetEvaluationOutput), (
        f"Expected {MMDetEvaluationOutput}, got {type(model_output)}"
    )
    image_ids: list[int] = []
    gt_instances: list[GroundTruthInstances] = []
    pred_instances: list[PredInstances] = []

    for batch_sample in model_output.det_data_samples:
        scale_factor = batch_sample.metainfo.get("scale_factor")
        if "gt_instances" in batch_sample:
            batch_sample.gt_instances.bboxes = scale_boxes(
                batch_sample.gt_instances.bboxes, [1 / s for s in scale_factor]
            )

        image_ids.append(batch_sample.metainfo["img_id"])
        gt_instances.append(
            GroundTruthInstances(
                bboxes=batch_sample.gt_instances["bboxes"],
                labels=batch_sample.gt_instances["labels"],
            )
        )
        pred_instances.append(
            PredInstances(
                bboxes=batch_sample.pred_instances["bboxes"],
                labels=batch_sample.pred_instances["labels"],
                scores=batch_sample.pred_instances["scores"],
            )
        )
    return image_ids, gt_instances, pred_instances
