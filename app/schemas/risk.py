from dataclasses import dataclass
from enum import Enum

@dataclass(slots=True)
class ConfidenceEstimate:

    mean: float

    minimum: float

    maximum: float

    std: float

    below_threshold: int

    confidence_level: str



@dataclass(slots=True)
class UncertaintyEstimate:

    epistemic: float

    aleatoric: float

    entropy: float

    total: float

@dataclass(slots=True)
class FailureReport:

    confidence_failures: int

    uncertainty_failures: int

    localization_failures: int

    total_failures: int

    status: str


@dataclass(slots=True)
class RiskAssessment:

    score: float

    level: str

    confidence: float

    uncertainty: float

    failures: int

@dataclass(slots=True)
class DecisionResult:

    risk_score: float

    risk_level: str

    decision: str

    recommended_speed: int

    warning: str

    recommendation: str

class SafetyDecision(Enum):    
    CONTINUE = "CONTINUE"
    PROCEED_WITH_CAUTION = "PROCEED_WITH_CAUTION"
    REDUCE_SPEED = "REDUCE_SPEED"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass(slots=True)
class RiskAnalysis:

    confidence: ConfidenceEstimate

    uncertainty: UncertaintyEstimate

    failures: FailureReport

    assessment: RiskAssessment

    decision: DecisionResult