import collections
import json
import random
from collections.abc import Iterable
from pathlib import Path

from atria_types import (
    AnnotatedObject,
    BoundingBox,
    BoundingBoxMode,
    ClassificationAnnotation,
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    Image,
    Label,
    LayoutAnalysisAnnotation,
)
from atria_types._generic._doc_content import DocumentContent
from atria_types._generic._pdf import PDF

from atria_datasets import DATASETS, DatasetConfig, DocumentDataset

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{doclaynet2022,
  title = {DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis},
  doi = {10.1145/3534678.353904},
  url = {https://arxiv.org/abs/2206.01062},
  author = {Pfitzmann, Birgit and Auer, Christoph and Dolfi, Michele and Nassar, Ahmed S and Staar, Peter W J},
  year = {2022}
}
"""
_DESCRIPTION = """\
DocLayNet is a human-annotated document layout segmentation dataset from a broad variety of document sources.
"""
_HOMEPAGE = "https://developer.ibm.com/exchanges/data/all/doclaynet/"
_LICENSE = "CDLA-Permissive-1.0"

_DATA_URLS = {
    "DocLayNet_core": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip",
    "DocLayNet_extras": "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_extra.zip",
}
_DOC_CLASSES = [
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
]

_LAYOUT_CLASSES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title ",
]


class SplitIterator:
    def __init__(self, split: DatasetSplitType, data_dir: str, seed: int = 42):
        core = Path(data_dir) / "DocLayNet_core"
        extras = Path(data_dir) / "DocLayNet_extras"
        split_files = {
            DatasetSplitType.train: "train.json",
            DatasetSplitType.validation: "val.json",
            DatasetSplitType.test: "test.json",
        }
        self.coco_path = core / "COCO" / split_files[split]
        self.image_dir = core / "PNG"
        self.pdf_path = extras / "PDF"
        self.seed = seed

    def __iter__(self):
        with open(self.coco_path, encoding="utf8") as f:
            annotation_data = json.load(f)
            images = annotation_data["images"]
            annotations = annotation_data["annotations"]
            image_id_to_annotations = collections.defaultdict(list)
            for annotation in annotations:
                image_id_to_annotations[annotation["image_id"]].append(annotation)

        random.seed(self.seed)
        random.shuffle(images)

        for image_info in images:
            annotations = image_id_to_annotations[image_info["id"]]
            image_info["image_path"] = str(self.image_dir / image_info["file_name"])
            image_info["pdf_path"] = str(
                self.pdf_path / image_info["file_name"].replace(".png", ".pdf")
            )
            yield image_info, annotations


class DocLayNetConfig(DatasetConfig):
    dataset_name: str = "doclaynet"
    load_ocr: bool = False


@DATASETS.register(
    "doclaynet",
    configs={
        "default": DocLayNetConfig(config_name="default"),
        "1k": DocLayNetConfig(
            config_name="1k", max_train_samples=1000, max_validation_samples=1000
        ),
    },
)
class DocLayNet(DocumentDataset[DocLayNetConfig]):
    __config__ = DocLayNetConfig

    def _download_urls(self) -> dict[str, tuple[str, str]] | dict[str, str] | list[str]:
        return _DATA_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            citation=_CITATION,
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            dataset_labels=DatasetLabels(
                classification=_DOC_CLASSES, layout=_LAYOUT_CLASSES
            ),
        )

    def _available_splits(self):
        return [
            DatasetSplitType.train,
            DatasetSplitType.validation,
            DatasetSplitType.test,
        ]

    def _split_iterator(self, split: DatasetSplitType, data_dir: str) -> Iterable:
        return SplitIterator(split=split, data_dir=data_dir, seed=self.config.seed)

    def _read_objects(
        self, annotations: list[dict], image_width: int, image_height: int
    ) -> list[AnnotatedObject]:
        objects = []
        for annotation in annotations:
            if annotation.get("ignore", False):
                continue
            bbox = BoundingBox(
                value=annotation["bbox"], mode=BoundingBoxMode.XYWH
            ).ops.switch_mode()
            if annotation["area"] <= 0 or bbox.width < 1 or bbox.height < 1:
                continue
            category_idx = annotation["category_id"] - 1
            if category_idx < 0 or category_idx > len(_LAYOUT_CLASSES):
                raise ValueError(
                    f"Invalid category_id {annotation['category_id']} in annotation."
                )
            if bbox.x2 > image_width or bbox.y2 > image_height:
                bbox = BoundingBox(
                    value=[
                        min(bbox.x1, image_width - 1),
                        min(bbox.y1, image_height - 1),
                        min(bbox.x2, image_width - 1),
                        min(bbox.y2, image_height - 1),
                    ],
                    mode=BoundingBoxMode.XYXY,
                )
            objects.append(
                AnnotatedObject(
                    label=Label(value=category_idx, name=_LAYOUT_CLASSES[category_idx]),
                    bbox=bbox.ops.normalize(width=image_width, height=image_height),
                    segmentation=annotation["segmentation"],
                    iscrowd=bool(annotation["iscrowd"]),
                )
            )
        return objects

    def _input_transform(self, sample: tuple[dict, list[dict]]) -> DocumentInstance:
        image_info, annotations = sample
        annotated_objects = self._read_objects(
            annotations, image_info["width"], image_info["height"]
        )
        pdf = PDF(file_path=image_info["pdf_path"]).load()
        assert pdf.num_pages is not None and pdf.num_pages == 1, (
            "Each PDF should have exactly one page."
        )
        text_elements = pdf.extract_text_elements(page_number=0)

        return DocumentInstance(
            sample_id=image_info["file_name"],
            image=Image(file_path=image_info["image_path"]),
            content=DocumentContent(text_elements=text_elements),
            annotations=[
                ClassificationAnnotation(
                    label=Label(
                        value=_DOC_CLASSES.index(image_info["doc_category"]),
                        name=image_info["doc_category"],
                    )
                ),
                LayoutAnalysisAnnotation(annotated_objects=annotated_objects),
            ],
        )
