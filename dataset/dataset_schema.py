from typing import List
from pydantic import BaseModel, field_validator
import ast

from enum import Enum

#----------- Enum --------


class RelationSense(str, Enum):
    EXPANSION_CONJUNCTION = "Expansion.Conjunction"
    EXPANSION_INSTANTIATION = "Expansion.Instantiation"
    EXPANSION_ALTERNATIVE = "Expansion.Alternative"
    EXPANSION_EQUIVALENCE = "Expansion.Equivalence"

    CONTINGENCY_CAUSE = "Contingency.Cause"
    CONTINGENCY_CONDITION = "Contingency.Condition"
    CONTINGENCY_PURPOSE = "Contingency.Purpose"

    COMPARISON_CONTRAST = "Comparison.Contrast"
    COMPARISON_CONCESSION = "Comparison.Concession"
    CONTINGENCY_SIMILARITY = "Comparison.Similarity"

    TEMPORAL_ASYNCHRONOUS = "Temporal.Asynchronous"
    TEMPORAL_SYNCHRONOUS = "Temporal.Synchrony"

class TamilConnectiveType(str, Enum):
    LEXICAL_FREE_TOKEN = "lexical"
    DC_SUFFIX = "suffixal"

class DiscourseRelationType(str, Enum):
    EXPLICIT = "Explicit"
    IMPLICIT = "Implicit"


# ---------- Basic Components ----------

class SpanText(BaseModel):
    raw_text: str
    span: List[int]


class Morphology(BaseModel):
    stem: str
    suffix: str


class TamilConnective(BaseModel):
    raw_text: str
    span: List[int]
    type: TamilConnectiveType
    morphology: Morphology


class EnglishConnective(BaseModel):
    raw_text: str
    span: List[int]


class Relation(BaseModel):
    type: DiscourseRelationType
    sense: RelationSense

    @field_validator("sense", mode="before")
    @classmethod
    def parse_sense(cls, v):
        # If it's like "['Expansion.Conjunction']"
        if isinstance(v, str) and v.startswith("["):
            parsed = ast.literal_eval(v) # parse string into a list
            return parsed[0]

        return v


# ---------- Language Block ----------

class EnglishEntry(BaseModel):
    sentence: str
    arg1: SpanText
    arg2: SpanText
    connective: EnglishConnective
    relation: Relation


class TamilEntry(BaseModel):
    sentence: str
    arg1: SpanText
    arg2: SpanText
    connective: TamilConnective
    relation: Relation


# ---------- Main Dataset Entry ----------

class DatasetEntry(BaseModel):
    id: int
    english: EnglishEntry
    tamil: TamilEntry
