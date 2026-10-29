"""Request/response models. Ranges match the data-cleaning rules exactly."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PatientInput(BaseModel):
    age_years: float = Field(..., ge=18, le=100, description="Age in years")
    gender: Literal[1, 2] = Field(..., description="1 = female, 2 = male (dataset coding)")
    height: float = Field(..., ge=120, le=220, description="cm")
    weight: float = Field(..., ge=30, le=200, description="kg")
    ap_hi: float = Field(..., ge=80, le=250, description="Systolic blood pressure, mmHg")
    ap_lo: float = Field(..., ge=40, le=160, description="Diastolic blood pressure, mmHg")
    cholesterol: Literal[1, 2, 3] = Field(..., description="1 normal, 2 above normal, 3 well above")
    gluc: Literal[1, 2, 3] = Field(..., description="1 normal, 2 above normal, 3 well above")
    smoke: Literal[0, 1]
    alco: Literal[0, 1]
    active: Literal[0, 1]

    @model_validator(mode="after")
    def check_bp(self):
        if self.ap_lo >= self.ap_hi:
            raise ValueError("ap_lo (diastolic) must be lower than ap_hi (systolic)")
        return self

    model_config = {"json_schema_extra": {"example": {
        "age_years": 52, "gender": 2, "height": 172, "weight": 84, "ap_hi": 140,
        "ap_lo": 90, "cholesterol": 2, "gluc": 1, "smoke": 0, "alco": 0, "active": 1}}}


class PredictionOut(BaseModel):
    request_id: str
    probability: float
    prediction: int
    risk_label: str
    model_version: str


class Contribution(BaseModel):
    label: str
    value: str          # the patient's value, formatted for display
    shap: float         # + raises risk, - lowers risk


class GroupContribution(Contribution):
    group: str
    members: list[str]


class FeatureContribution(Contribution):
    feature: str


class Explanation(BaseModel):
    output_space: Literal["log_odds", "probability"]
    base_value: float          # model's average output on the training data
    base_probability: float    # the same, as a probability
    output_value: float        # base_value + sum of SHAP values
    groups: list[GroupContribution]       # related measurements combined (for people)
    features: list[FeatureContribution]   # every model input (for analysis)


class ExplainedPredictionOut(PredictionOut):
    explanation: Explanation


class BatchIn(BaseModel):
    patients: list[PatientInput] = Field(..., min_length=1)


class BatchOut(BaseModel):
    model_version: str
    predictions: list[PredictionOut]


class LabelIn(BaseModel):
    """Ground truth arriving later (e.g. doctor-confirmed diagnosis)."""
    label: Literal[0, 1]
    label_source: Literal["human"] = "human"
