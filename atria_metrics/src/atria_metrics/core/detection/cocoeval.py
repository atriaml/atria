import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import ignite.distributed as idist
import torch
from atria_models.core.types.model_outputs import MMDetEvaluationOutput
from ignite.engine import Engine
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def xyxy2xywh(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


@dataclass
class CocoInstance:
    bboxes: torch.Tensor
    labels: torch.Tensor

    def detach(self):
        self.bboxes = self.bboxes.detach()
        self.labels = self.labels.detach()


@dataclass
class GroundTruthInstance(CocoInstance):
    pass


@dataclass
class PredInstance(CocoInstance):
    scores: torch.Tensor

    def detach(self):
        super().detach()
        self.scores = self.scores.detach()


@dataclass
class CocoCategory:
    id: int
    name: str
    supercategory: str


def _cocoeval_output_transform(output: MMDetEvaluationOutput):
    from mmdet.structures.bbox import scale_boxes

    assert isinstance(output, MMDetEvaluationOutput), (
        f"Expected {MMDetEvaluationOutput}, got {type(output)}"
    )
    assert output.det_data_samples is not None, (
        "output.det_data_samples is None. Cannot transform for COCOEvalMetric."
    )
    image_ids: list[int] = []
    gt_instances: list[GroundTruthInstance] = []
    pred_instances: list[PredInstance] = []

    for batch_sample in output.det_data_samples:
        scale_factor = batch_sample.metainfo.get("scale_factor")
        if "gt_instances" in batch_sample:
            batch_sample.gt_instances.bboxes = scale_boxes(
                batch_sample.gt_instances.bboxes,
                (1 / scale_factor[0], 1 / scale_factor[1]),
            )

        image_ids.append(batch_sample.metainfo["img_id"])
        gt_instances.append(
            GroundTruthInstance(
                bboxes=batch_sample.gt_instances["bboxes"],
                labels=batch_sample.gt_instances["labels"],
            )
        )
        pred_instances.append(
            PredInstance(
                bboxes=batch_sample.pred_instances["bboxes"],
                labels=batch_sample.pred_instances["labels"],
                scores=batch_sample.pred_instances["scores"],
            )
        )
    return image_ids, gt_instances, pred_instances


class COCOEvalMetric(Metric):
    """COCO-style evaluation metric for object detection.

    This class accumulates only essential data and performs COCO evaluation
    after gathering the results from all GPUs during `compute()`.

    Args:
        device: The device where internal storage will reside.
    """

    _state_dict_all_req_keys = ("_image_ids", "_gt_instances", "_pred_instances")
    _output_keys = (
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
    )

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._image_ids: list[int] = []
        self._gt_instances: list[GroundTruthInstance] = []
        self._pred_instances: list[PredInstance] = []
        self._result: dict[str, float] | None = None

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        image_ids, gt_instances, pred_instances = output
        self.update(image_ids, gt_instances, pred_instances)

    def _check_type(
        self,
        image_ids: list[int],
        gt_instances: list[GroundTruthInstance],
        pred_instances: list[PredInstance],
    ) -> None:
        if not isinstance(image_ids, list) or not all(
            isinstance(i, int) for i in image_ids
        ):
            raise TypeError("image_ids must be a list of integers.")
        if not isinstance(gt_instances, list) or not all(
            isinstance(gt, GroundTruthInstance) for gt in gt_instances
        ):
            raise TypeError("gt_instances must be a list of GroundTruthInstance.")
        if not isinstance(pred_instances, list) or not all(
            isinstance(pred, PredInstance) for pred in pred_instances
        ):
            raise TypeError("pred_instances must be a list of PredInstance.")

    @reinit__is_reduced
    def update(
        self,
        image_ids: list[int],
        gt_instances: list[GroundTruthInstance],
        pred_instances: list[PredInstance],
    ):
        self._check_type(image_ids, gt_instances, pred_instances)
        for x in gt_instances:
            x.detach()
        for x in pred_instances:
            x.detach()
        self._image_ids.extend(image_ids)
        self._gt_instances.extend(gt_instances)
        self._pred_instances.extend(pred_instances)

    def compute(self) -> float:
        if len(self._image_ids) == 0:
            raise ValueError("No data available for COCO evaluation.")

        if self._result is None:
            ws = idist.get_world_size()
            if ws > 1:
                self._image_ids = idist.all_gather(self._image_ids)  # type: ignore
                self._gt_instances = idist.all_gather(self._gt_instances)  # type: ignore
                self._pred_instances = idist.all_gather(self._pred_instances)  # type: ignore

            if idist.get_rank() == 0:
                # Run compute_fn on zero rank only
                self._result = self._call_coco_eval(
                    self._image_ids, self._gt_instances, self._pred_instances
                )

            if ws > 1:
                # broadcast result to all processes
                self._result = idist.broadcast(self._result, src=0)  # type: ignore

        return self._result  # type: ignore

    def _from_instance_to_ann(
        self,
        image_ids: Sequence[int],
        instances: Sequence[PredInstance] | Sequence[GroundTruthInstance],
    ) -> list[dict]:
        # Create COCO format ground truth annotations
        anns = []
        for idx, instance in enumerate(instances):
            image_id = image_ids[idx]
            for i, bbox in enumerate(instance.bboxes):
                bbox = xyxy2xywh(bbox.tolist())
                ann = {
                    "id": len(anns) + 1,
                    "image_id": image_id,
                    "category_id": int(instance.labels[i])
                    + 1,  # COCO categories start at 1
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],  # width * height
                    "iscrowd": 0,
                }
                if hasattr(instance, "scores"):
                    ann["score"] = float(instance.scores[i])  # type: ignore
                anns.append(ann)
        return anns

    def _call_coco_eval(
        self,
        image_ids: list[int],
        gt_instances: list[GroundTruthInstance],
        pred_instances: list[PredInstance],
    ) -> dict[str, float]:
        coco_gt_annotations = self._from_instance_to_ann(
            image_ids=image_ids, instances=gt_instances
        )
        coco_pred_annotations = self._from_instance_to_ann(
            image_ids=image_ids, instances=pred_instances
        )

        # If there are no predictions, return zeros to prevent evaluation errors
        if len(coco_pred_annotations) == 0:
            return dict.fromkeys(self._output_keys, 0.0)

        # Initialize COCO ground truth and result annotations
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": image_id} for image_id in image_ids],
            "annotations": coco_gt_annotations,  # type: ignore
            "categories": [{"id": 1, "name": "table"}],
            "info": {},
        }
        with open("coco_gt.json", "w") as f:
            json.dump(coco_gt.dataset, f)
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(coco_pred_annotations)  # type: ignore
        with open("coco_dt.json", "w") as f:
            json.dump(coco_dt.dataset, f)

        # Perform evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return {key: coco_eval.stats[i] for i, key in enumerate(self._output_keys)}

    def completed(self, engine: Engine, name: str) -> None:
        result = self.compute()
        if isinstance(result, Mapping):
            if name in result.keys():
                raise ValueError(
                    f"Argument name '{name}' is conflicting with mapping keys: {list(result.keys())}"
                )

            for key, value in result.items():
                engine.state.metrics[name + "_" + key] = value
        else:
            if isinstance(result, torch.Tensor):
                if len(result.size()) == 0:
                    result = result.item()
                elif "cpu" not in result.device.type:
                    result = result.cpu()

            engine.state.metrics[name] = result
