import os
import random
from collections.abc import Generator, Iterable
from pathlib import Path

from atria_logger import get_logger
from atria_types import (
    DatasetLabels,
    DatasetMetadata,
    DatasetSplitType,
    DocumentInstance,
    Image,
    LayoutAnalysisAnnotation,
)
from atria_types._generic._doc_content import DocumentContent

from atria_datasets import DATASETS, DocumentDataset
from atria_datasets.core.dataset._datasets import DatasetConfig

from .utilities import read_pascal_voc, read_words_json

logger = get_logger(__name__)

_CITATION = """\
@software{smock2021tabletransformer,
    author = {Smock, Brandon and Pesala, Rohith},
    month = {06},
    title = {{Table Transformer}},
    url = {https://github.com/microsoft/table-transformer},
    version = {1.0.0},
    year = {2021}
}
"""

_DESCRIPTION = """\
PubTables-1M: Towards comprehensive table extraction from unstructured documents.
"""

_HOMEPAGE = "https://github.com/microsoft/table-transformer/"

_LICENSE = "https://github.com/microsoft/table-transformer/blob/main/LICENSE"

_DETECTION_URLS = [
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Annotations_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Annotations_Train.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Annotations_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Filelists.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Train_Part1.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Train_Part2.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Images_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Detection_Page_Words.tar.gz",
]

_DETECTION_LABELS = ["table", "table rotated"]

_STRUCTURE_URLS = [
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Annotations_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Annotations_Train.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Annotations_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Filelists.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Images_Test.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Images_Train.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Images_Val.tar.gz",
    "https://huggingface.co/datasets/bsmock/pubtables-1m/resolve/main/PubTables-1M-Structure_Table_Words.tar.gz",
]

_STRUCTURE_LABELS = [
    "table",
    "table column",
    "table row",
    "table column header",
    "table projected row header",
    "table spanning cell",
]


def folder_iterator(folder):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            yield os.path.join(subdir, file)


class SplitIterator:
    def __init__(
        self, task: str, data_dir: str, split: DatasetSplitType, seed: int = 42
    ) -> None:
        self.split = split
        self.data_dir = Path(data_dir)
        self.task = task

        # Get file lists
        if split == DatasetSplitType.test:
            split_name = "test"
        elif split == DatasetSplitType.validation:
            split_name = "val"
        elif split == DatasetSplitType.train:
            split_name = "train"

        self.xml_filelist = (
            self.data_dir
            / f"PubTables-1M-{self.task.capitalize()}_Filelists"
            / f"{split_name}_filelist.txt"
        )
        self.images_filelist = (
            self.data_dir
            / f"PubTables-1M-{self.task.capitalize()}_Filelists"
            / "images_filelist.txt"
        )

        self.ann_dir = (
            self.data_dir
            / f"PubTables-1M-{self.task.capitalize()}_Annotations_{split_name.capitalize()}"
        )
        self.img_dir = (
            self.data_dir
            / f"PubTables-1M-{self.task.capitalize()}_Images_{split_name.capitalize()}"
        )
        self.words_dir = self.data_dir / "PubTables-1M-Structure_Table_Words"
        self.seed = seed

        if not self.ann_dir.exists():
            raise FileNotFoundError(
                f"Annotation directory {self.ann_dir} does not exist."
            )
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory {self.img_dir} does not exist.")

    def __iter__(self) -> Generator[tuple[Path, Path, Path], None, None]:
        # Read XML file list
        with open(self.xml_filelist) as file:
            lines = file.readlines()
            lines = [l.split("/")[-1].strip() for l in lines]
        xml_file_names = {
            f.strip().replace(".xml", "") for f in lines if f.strip().endswith(".xml")
        }

        # Read images file list
        with open(self.images_filelist) as file:
            lines = file.readlines()
        image_file_paths = {
            f.strip().replace(".jpg", "").replace("images/", "")
            for f in lines
            if f.strip().endswith(".jpg")
        }

        file_paths = sorted(xml_file_names.intersection(image_file_paths))

        random.seed(self.seed)
        random.shuffle(file_paths)

        logger.info(f"Generating {len(file_paths)} samples...")
        for sample_file_path in file_paths:
            ann_file = self.ann_dir / (sample_file_path + ".xml")
            img_file = self.img_dir / (sample_file_path + ".jpg")
            word_file = self.words_dir / (sample_file_path + "_words.json")
            if not ann_file.exists() or not img_file.exists():
                logger.warning(
                    f"Skipping {sample_file_path} because image or annotation file does not exist."
                )
                continue

            yield img_file, ann_file, word_file

    def __len__(self) -> int:
        with open(self.xml_filelist) as file:
            lines = file.readlines()
        return len([line for line in lines if line.strip().endswith(".xml")])


class PubTables1MConfig(DatasetConfig):
    dataset_name: str = "pubtables1m"
    task: str = "structure"  # "structure" or "detection"


@DATASETS.register(
    "pubtables1m",
    configs={
        "detection": PubTables1MConfig(config_name="detection", task="detection"),
        "detection_1k": PubTables1MConfig(
            config_name="detection_1k",
            task="detection",
            max_train_samples=1000,
            max_validation_samples=1000,
        ),
        "structure": PubTables1MConfig(config_name="structure", task="structure"),
        "structure_1k": PubTables1MConfig(
            config_name="structure_1k",
            task="structure",
            max_train_samples=1000,
            max_validation_samples=1000,
        ),
    },
)
class PubTables1M(DocumentDataset[PubTables1MConfig]):
    __config__ = PubTables1MConfig

    def _download_urls(self) -> list[str]:
        return _STRUCTURE_URLS if self.config.task == "structure" else _DETECTION_URLS

    def _metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            dataset_labels=DatasetLabels(
                layout=_STRUCTURE_LABELS
                if self.config.task == "structure"
                else _DETECTION_LABELS
            ),
        )

    def _available_splits(self) -> list[DatasetSplitType]:
        return [
            DatasetSplitType.train,
            DatasetSplitType.validation,
            DatasetSplitType.test,
        ]

    def _split_iterator(self, split: DatasetSplitType, data_dir: str) -> Iterable:
        return SplitIterator(
            task=self.config.task, split=split, data_dir=data_dir, seed=self.config.seed
        )

    def _input_transform(self, inputs: tuple[Path, Path, Path]) -> DocumentInstance:
        image_file_path, xml_filepath, word_file_path = inputs
        image = Image(file_path=str(image_file_path)).load()
        annotated_objects = read_pascal_voc(
            str(xml_filepath),
            labels=(
                _STRUCTURE_LABELS
                if self.config.task == "structure"
                else _DETECTION_LABELS
            ),
            image_width=image.width,
            image_height=image.height,
        )
        text_elements = read_words_json(
            str(word_file_path), image_width=image.width, image_height=image.height
        )
        return DocumentInstance(
            sample_id=Path(image_file_path).name,
            image=image,
            content=DocumentContent(text_elements=text_elements),
            annotations=[LayoutAnalysisAnnotation(annotated_objects=annotated_objects)],
        )
