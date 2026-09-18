"""Typed records shared by the hierarchical TransWeave stages."""

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class SourceFrameKey:
    """A frame identity qualified by its source SENF."""

    source_id: str
    frame_id: str


@dataclass(frozen=True)
class PairComponentCosts:
    structural: float = 0.0
    exemplar: float = 0.0
    identity: float = 0.0
    conflict: float = 0.0
    time: float = 0.0
    location: float = 0.0
    modality: float = 0.0
    branch: float = 0.0
    temporal_decay: float = 0.0
    persistence: float = 0.0

    @property
    def total(self) -> float:
        return sum((
            self.structural,
            self.exemplar,
            self.identity,
            self.conflict,
            self.time,
            self.location,
            self.modality,
            self.branch,
            self.temporal_decay,
            self.persistence,
        ))


@dataclass(frozen=True)
class FrameAlignment:
    query_frame_id: str
    source_frame: SourceFrameKey
    score: float
    costs: PairComponentCosts
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolishStageDiagnostics:
    solves: int = 0
    iterations: int = 0
    converged: bool = True
    row_residual: float = 0.0
    column_residual: float = 0.0
    global_residual: float = 0.0
    objective: float = 0.0
    entropy: float = 0.0


@dataclass(frozen=True)
class PolishDiagnostics:
    engine: str = "beam"
    local: PolishStageDiagnostics = PolishStageDiagnostics()
    block: PolishStageDiagnostics = PolishStageDiagnostics()
    global_stage: PolishStageDiagnostics = PolishStageDiagnostics()
    candidate_count: int = 0
    seed_count: int = 0
    fallback_reason: str = ""
    matched_soft_mass: float = 0.0
    unmatched_soft_mass: float = 0.0

    @property
    def residuals(self) -> tuple[float, float, float]:
        return (
            self.local.global_residual,
            self.block.global_residual,
            self.global_stage.global_residual,
        )

    @property
    def global_coupling_objective(self) -> float:
        return self.global_stage.objective

    @property
    def entropy(self) -> float:
        return self.global_stage.entropy

    @property
    def iterations(self) -> int:
        return self.local.iterations + self.block.iterations + self.global_stage.iterations

    @property
    def converged(self) -> bool:
        return (
            not self.fallback_reason
            and self.local.converged
            and self.block.converged
            and self.global_stage.converged
        )


__all__ = [
    "FrameAlignment",
    "PairComponentCosts",
    "PolishDiagnostics",
    "PolishStageDiagnostics",
    "SourceFrameKey",
]
