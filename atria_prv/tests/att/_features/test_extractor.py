from __future__ import annotations

import pytest
import torch

from atria_prv.att._features._bio_scheme import BioScheme
from atria_prv.att._features._extractor import AggConfig, TokenSignalExtractor
from atria_prv.att._features._factory import build_feature_extractor
from atria_prv.att._features._signals import SIGNAL_FUNCS

IGNORE_INDEX = -100
MAX_LENGTH = 20
LABEL_NAMES = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]

# a handful of hand-labeled FUNSD-style documents; the last one has no entities at all,
# to exercise the empty-split NaN-safe fill path.
SAMPLE_DOCUMENTS = [
    (
        ["Invoice", "Date", ":", "March", "3", "2020"],
        ["B-HEADER", "I-HEADER", "O", "B-ANSWER", "I-ANSWER", "I-ANSWER"],
    ),
    (
        ["Company", "Name", "Acme", "Corp"],
        ["B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"],
    ),
    (
        ["Total", "Due", "$", "42", "."],
        ["B-QUESTION", "I-QUESTION", "O", "B-ANSWER", "O"],
    ),
    (
        ["Please", "sign", "below", "."],
        ["O", "O", "O", "O"],
    ),
]


@pytest.fixture(scope="module")
def bert_token_classifier():
    """A real (untrained-head) BERT token-classification model + tokenizer."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model_name = "bert-base-uncased"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(LABEL_NAMES)
        )
    except OSError as e:
        pytest.skip(f"transformers model '{model_name}' unavailable (offline?): {e}")
    model.eval()
    return tokenizer, model


def _build_batch(tokenizer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize the sample documents and align word-level BIO labels to subtokens."""
    all_input_ids, all_attention_mask, all_labels = [], [], []
    for words, word_labels in SAMPLE_DOCUMENTS:
        encoded = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = encoded.word_ids(batch_index=0)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(IGNORE_INDEX)
            elif word_id != prev_word_id:
                label_ids.append(LABEL_NAMES.index(word_labels[word_id]))
            else:
                # only the first subtoken of a word carries the label
                label_ids.append(IGNORE_INDEX)
            prev_word_id = word_id

        all_input_ids.append(encoded["input_ids"][0])
        all_attention_mask.append(encoded["attention_mask"][0])
        all_labels.append(torch.tensor(label_ids))

    return (
        torch.stack(all_input_ids),
        torch.stack(all_attention_mask),
        torch.stack(all_labels),
    )


def test_build_feature_extractor_dispatch():
    """build_feature_extractor dispatches via isinstance(model_pipeline, TokenClassificationPipeline)."""
    from types import SimpleNamespace

    from atria_models.core.model_pipelines import (
        LayoutTokenClassificationPipeline,
        TokenClassificationPipeline,
    )

    labels = SimpleNamespace(ser=LABEL_NAMES)

    # __new__ bypasses __init__ (no config/model build needed) but still satisfies isinstance;
    # _labels is set manually since __init__ (which would normally set it) never runs.
    token_classification_pipeline = TokenClassificationPipeline.__new__(
        TokenClassificationPipeline
    )
    token_classification_pipeline._labels = labels
    extractor = build_feature_extractor(token_classification_pipeline)
    assert isinstance(extractor, TokenSignalExtractor)

    # LayoutTokenClassificationPipeline subclasses TokenClassificationPipeline, so it's
    # accepted too — an intentional simplification, not an oversight.
    layout_pipeline = LayoutTokenClassificationPipeline.__new__(
        LayoutTokenClassificationPipeline
    )
    layout_pipeline._labels = labels
    assert isinstance(build_feature_extractor(layout_pipeline), TokenSignalExtractor)

    with pytest.raises(ValueError, match="Unsupported model pipeline type"):
        build_feature_extractor(object())


def test_to_features_on_real_bert_logits(bert_token_classifier):
    """Run real BERT logits through TokenSignalExtractor and inspect the result."""
    tokenizer, model = bert_token_classifier
    input_ids, attention_mask, labels = _build_batch(tokenizer)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    bio_scheme = BioScheme.from_label_names(LABEL_NAMES)
    extractor = TokenSignalExtractor(
        signals=SIGNAL_FUNCS,
        bio_scheme=bio_scheme,
        config=AggConfig(),
        num_labels=len(LABEL_NAMES),
    )
    df = extractor.to_features(logits, labels)

    assert len(df) == len(SAMPLE_DOCUMENTS)
    assert not df.isna().any().any()

    # the last sample document has no entity tokens at all
    assert df.loc[3, "loss__span_start__count"] == 0
    assert df.loc[3, "loss__span_start__mean"] == AggConfig().empty_split_fill

    for signal in SIGNAL_FUNCS:
        assert f"{signal}__all__mean" in df.columns
        assert f"{signal}__entity__count" in df.columns

    # printed for manual inspection with `pytest -s`
    summary_columns = [c for c in df.columns if c.endswith("__mean") or c.endswith("__count")]
    print("\nTokenSignalExtractor features on real BERT logits:")
    print(df[summary_columns].to_string())
