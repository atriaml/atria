from atria_types._base._data_model import BaseDataModel
from atria_types._pydantic import (
    IntField,
    StrField,
)


class AnswerSpan(BaseDataModel):
    start: IntField
    end: IntField
    text: StrField


class QAPair(BaseDataModel):
    id: IntField
    question_text: StrField
    answer_spans: list[AnswerSpan]
