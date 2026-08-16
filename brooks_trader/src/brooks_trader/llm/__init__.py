"""Read-only LLM explanation contracts isolated from trading and execution."""

from brooks_trader.llm.explainer import (
    BrooksReference,
    DecisionExplanation,
    DetectedSetupExplanation,
    ExplanationNarrative,
    ExplanationRequest,
    ExplanationResult,
    ExplanationStatus,
    HistoricalStatisticsExplanation,
    LLMExplainer,
    LLMProvider,
    MarketContextExplanation,
    ProbabilityStatus,
    RiskRewardExplanation,
    TradeExplanation,
    WarningExplanation,
    build_explanation_prompts,
)

__all__ = [
    "BrooksReference",
    "DecisionExplanation",
    "DetectedSetupExplanation",
    "ExplanationNarrative",
    "ExplanationRequest",
    "ExplanationResult",
    "ExplanationStatus",
    "HistoricalStatisticsExplanation",
    "LLMExplainer",
    "LLMProvider",
    "MarketContextExplanation",
    "ProbabilityStatus",
    "RiskRewardExplanation",
    "TradeExplanation",
    "WarningExplanation",
    "build_explanation_prompts",
]
