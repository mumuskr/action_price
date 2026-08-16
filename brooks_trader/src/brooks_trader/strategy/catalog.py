"""The strategy module catalog and transparent module selection contract.

The catalog intentionally separates a Brooks concept from the project's executable
interpretation.  A module can therefore be inspected in the dashboard without
presenting a research proxy as an exact formula from the books.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ModuleStatus(StrEnum):
    """Implementation state exposed to the strategy library."""

    IMPLEMENTED = "implemented"
    PROXY = "proxy"
    PLANNED = "planned"


class StrategyModuleSelection(BaseModel):
    """Boolean switches for modules currently supported by the backtester."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    h2_with_trend: bool = True
    l2_with_trend: bool = True
    market_regime_filter: bool = True
    ema_alignment_filter: bool = True
    pattern_quality_filter: bool = True
    context_quality_filter: bool = True
    signal_bar_filter: bool = True
    pressure_filter: bool = True
    pullback_depth_filter: bool = True
    tight_trading_range_filter: bool = True
    recent_climax_filter: bool = True
    room_to_target_filter: bool = True

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any] | None = None,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> StrategyModuleSelection:
        """Resolve YAML values and runtime overrides without mutating either source."""
        merged = dict(values or {})
        merged.update(overrides or {})
        return cls.model_validate(merged)

    def enabled_ids(self) -> tuple[str, ...]:
        """Return enabled executable module IDs in stable catalog order."""
        values = self.model_dump()
        return tuple(module.id for module in STRATEGY_MODULES if values.get(module.id, False))


@dataclass(frozen=True)
class StrategyModule:
    """Human-readable module metadata used by the UI and experiment records."""

    id: str
    name: str
    category: str
    status: ModuleStatus
    book_concept: str
    description: str
    interpretation: str
    code_path: str
    default_enabled: bool = False


STRATEGY_MODULES: tuple[StrategyModule, ...] = (
    StrategyModule(
        id="h2_with_trend",
        name="H2 顺势多头",
        category="二次入场",
        status=ModuleStatus.IMPLEMENTED,
        book_concept="Second entry long in a bull trend",
        description="在多头背景的回调中识别 H1 后的第二次向上突破尝试。",
        interpretation="使用 H1/H2 状态机; 触发价为信号 K 线高点上方一个最小跳动。",
        code_path="src/brooks_trader/patterns/h1_h2.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="l2_with_trend",
        name="L2 顺势空头",
        category="二次入场",
        status=ModuleStatus.IMPLEMENTED,
        book_concept="Second entry short in a bear trend",
        description="在空头背景的反弹中识别 L1 后的第二次向下突破尝试。",
        interpretation="使用 L1/L2 状态机; 触发价为信号 K 线低点下方一个最小跳动。",
        code_path="src/brooks_trader/patterns/l1_l2.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="market_regime_filter",
        name="市场状态过滤",
        category="市场背景",
        status=ModuleStatus.PROXY,
        book_concept="Trend, trading range, and Always In context",
        description="只在与交易方向一致的趋势或 Always In 背景下接受 setup。",
        interpretation="市场状态由 EMA、结构、压力、重叠和突破五个分数组合得到。",
        code_path="src/brooks_trader/strategy/setup_engine.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="ema_alignment_filter",
        name="EMA 方向过滤",
        category="市场背景",
        status=ModuleStatus.PROXY,
        book_concept="Moving average context",
        description="要求价格位于 EMA 的对应方向, 且 EMA 斜率与交易方向一致。",
        interpretation="当前使用配置中的 EMA 周期和向后斜率; EMA 周期不是书中固定数字。",
        code_path="src/brooks_trader/strategy/setup_engine.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="pattern_quality_filter",
        name="形态质量过滤",
        category="信号质量",
        status=ModuleStatus.PROXY,
        book_concept="Quality of the second-entry pattern",
        description="过滤质量分低于最低阈值的 H2/L2 形态。",
        interpretation="质量分由背景、收盘位置和方向性实体按权重合成。",
        code_path="src/brooks_trader/patterns/base.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="context_quality_filter",
        name="背景质量过滤",
        category="信号质量",
        status=ModuleStatus.PROXY,
        book_concept="Trade with a sufficiently strong context",
        description="要求市场趋势分达到最低背景分。",
        interpretation="趋势分是可解释的数值代理, 不是书中给出的精确评分公式。",
        code_path="src/brooks_trader/strategy/setup_engine.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="signal_bar_filter",
        name="信号 K 线过滤",
        category="信号质量",
        status=ModuleStatus.PROXY,
        book_concept="Signal bar quality and close location",
        description="过滤实体过小或收盘位置不支持交易方向的信号 K 线。",
        interpretation="实体比例和收盘位置阈值可调, 属于程序化定义。",
        code_path="src/brooks_trader/strategy/setup_engine.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="pressure_filter",
        name="方向压力过滤",
        category="市场背景",
        status=ModuleStatus.PROXY,
        book_concept="Buying/selling pressure",
        description="过滤与交易方向相反或不足的近期方向压力。",
        interpretation="压力分由近期方向 K 线的滚动统计近似。",
        code_path="src/brooks_trader/strategy/setup_engine.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="pullback_depth_filter",
        name="回调深度过滤",
        category="市场背景",
        status=ModuleStatus.PROXY,
        book_concept="Pullback depth and trend quality",
        description="拒绝相对于平均 K 线范围过深的回调。",
        interpretation="以回调跨度除以平均范围计算深度, 阈值为项目参数。",
        code_path="src/brooks_trader/strategy/setup_engine.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="tight_trading_range_filter",
        name="紧密交易区间过滤",
        category="交易区间",
        status=ModuleStatus.PROXY,
        book_concept="Tight trading range",
        description="避免在重叠高、跨度窄的紧密交易区间内追入。",
        interpretation="使用窗口重叠均值和跨度/平均范围两个可调条件。",
        code_path="src/brooks_trader/strategy/setup_engine.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="recent_climax_filter",
        name="近期高潮过滤",
        category="交易区间",
        status=ModuleStatus.PROXY,
        book_concept="Climactic move and exhaustion",
        description="在近期出现多个方向性趋势 K 线后, 暂不接受新 setup。",
        interpretation="以回看窗口中方向趋势 K 线的数量近似高潮。",
        code_path="src/brooks_trader/strategy/setup_engine.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="room_to_target_filter",
        name="目标空间过滤",
        category="交易管理",
        status=ModuleStatus.PROXY,
        book_concept="Room to the target",
        description="如果历史支撑/阻力距离不足以覆盖目标 R 倍数, 则拒绝交易。",
        interpretation="在信号前的回看窗口查找最近的历史阻力或支撑。",
        code_path="src/brooks_trader/strategy/setup_engine.py",
        default_enabled=True,
    ),
    StrategyModule(
        id="breakout",
        name="突破",
        category="趋势与交易区间",
        status=ModuleStatus.PLANNED,
        book_concept="Breakout",
        description="识别交易区间或关键价位的突破 setup。",
        interpretation="检测器尚未接入回测管线。",
        code_path="src/brooks_trader/patterns/breakout.py",
    ),
    StrategyModule(
        id="failed_breakout",
        name="失败突破",
        category="反转",
        status=ModuleStatus.PLANNED,
        book_concept="Failed breakout",
        description="识别突破失败后反向入场的机会。",
        interpretation="检测器尚未接入回测管线。",
        code_path="src/brooks_trader/patterns/failed_breakout.py",
    ),
    StrategyModule(
        id="wedge",
        name="楔形",
        category="反转",
        status=ModuleStatus.PLANNED,
        book_concept="Wedge pattern",
        description="识别三推楔形及其反转信号。",
        interpretation="检测器尚未接入回测管线。",
        code_path="src/brooks_trader/patterns/wedge.py",
    ),
    StrategyModule(
        id="double_top_bottom",
        name="双顶/双底",
        category="反转",
        status=ModuleStatus.PLANNED,
        book_concept="Double top and double bottom",
        description="识别双顶、双底和对应的反转机会。",
        interpretation="检测器尚未接入回测管线。",
        code_path="src/brooks_trader/patterns/double_top_bottom.py",
    ),
    StrategyModule(
        id="major_trend_reversal",
        name="主要趋势反转",
        category="反转",
        status=ModuleStatus.PLANNED,
        book_concept="Major trend reversal",
        description="识别趋势衰竭、反转尝试和确认后的入场。",
        interpretation="检测器尚未接入回测管线。",
        code_path="src/brooks_trader/patterns/major_trend_reversal.py",
    ),
)


def get_strategy_module(module_id: str) -> StrategyModule:
    """Return one module by stable ID or raise a useful error."""
    for module in STRATEGY_MODULES:
        if module.id == module_id:
            return module
    raise KeyError(f"unknown strategy module: {module_id}")


def strategy_module_catalog() -> tuple[StrategyModule, ...]:
    """Return the immutable module catalog in display order."""
    return STRATEGY_MODULES
