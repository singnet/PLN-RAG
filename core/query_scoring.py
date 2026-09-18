from dataclasses import dataclass, field
from typing import Mapping, Optional


@dataclass(frozen=True)
class SENFSignals:
    """Weave-derived evidence about a question, plus the weights to score it with.

    Weights default to zero so that constructing this without configuring it
    reproduces the pre-SENF ranking exactly; the parser supplies real weights.
    """

    grounded_symbols: frozenset[str] = frozenset()
    role_signatures: frozenset[tuple[str, str]] = frozenset()
    distortion: float = 0.0
    source_grounding_weight: int = 0
    role_compat_weight: int = 0
    distortion_weight: int = 0
    identity_support: Mapping[str, float] = field(default_factory=dict)
    exemplar_coherence: Mapping[str, float] = field(default_factory=dict)
    conflict_penalty: Mapping[str, float] = field(default_factory=dict)
    transport_cost: float = 0.0
    identity_support_weight: int = 0
    exemplar_coherence_weight: int = 0
    conflict_weight: int = 0
    transport_cost_weight: int = 0
    matched_soft_mass: Optional[float] = None
    global_residual_ratio: Optional[float] = None
    alignment_confidence: Optional[float] = None
    matched_soft_mass_weight: int = 0
    global_residual_ratio_weight: int = 0
    alignment_confidence_weight: int = 0
    distortion_candidate_specific: bool = False

    @property
    def active(self) -> bool:
        return bool(
            self.source_grounding_weight
            or self.role_compat_weight
            or self.distortion_weight
            or self.identity_support_weight
            or self.exemplar_coherence_weight
            or self.conflict_weight
            or self.transport_cost_weight
            or self.matched_soft_mass_weight
            or self.global_residual_ratio_weight
            or self.alignment_confidence_weight
        )

    def bonus(self, query: dict) -> int:
        """Additive adjustment for one candidate. Zero weights give zero."""
        return sum(self.bonus_breakdown(query).values())

    def bonus_breakdown(self, query: dict) -> dict[str, int]:
        """Expose candidate-local SENF adjustments without changing their sum."""
        parts = {
            "source_grounding": 0,
            "role_compat": 0,
            "identity_support": 0,
            "exemplar_coherence": 0,
            "conflict": 0,
            "transport": 0,
            "distortion": 0,
            "matched_soft_mass": 0,
            "global_residual_ratio": 0,
            "alignment_confidence": 0,
        }
        if not self.active:
            return parts

        constants = [arg for arg in query["args"] if not arg.startswith(("$", "?"))]

        if self.source_grounding_weight and constants:
            if all(arg in self.grounded_symbols for arg in constants):
                parts["source_grounding"] = self.source_grounding_weight

        if self.role_compat_weight and self.role_signatures:
            if any((query["head"], arg) in self.role_signatures for arg in constants):
                parts["role_compat"] = self.role_compat_weight

        if constants and self.identity_support_weight:
            support = sum(self.identity_support.get(arg, 0.0) for arg in constants) / len(constants)
            parts["identity_support"] = round(support * self.identity_support_weight)

        if constants and self.exemplar_coherence_weight:
            coherence = sum(
                self.exemplar_coherence.get(arg, 0.0) for arg in constants
            ) / len(constants)
            parts["exemplar_coherence"] = round(
                coherence * self.exemplar_coherence_weight
            )

        if constants and self.conflict_weight:
            conflict = sum(self.conflict_penalty.get(arg, 0.0) for arg in constants) / len(constants)
            parts["conflict"] = -round(conflict * self.conflict_weight)

        if self.transport_cost_weight:
            parts["transport"] = -round(
                self.transport_cost * self.transport_cost_weight
            )

        if self.matched_soft_mass_weight and self.matched_soft_mass is not None:
            parts["matched_soft_mass"] = round(
                self.matched_soft_mass * self.matched_soft_mass_weight
            )
        if (
            self.global_residual_ratio_weight
            and self.global_residual_ratio is not None
        ):
            parts["global_residual_ratio"] = -round(
                self.global_residual_ratio * self.global_residual_ratio_weight
            )
        if self.alignment_confidence_weight and self.alignment_confidence is not None:
            parts["alignment_confidence"] = round(
                self.alignment_confidence * self.alignment_confidence_weight
            )

        # Question-global distortion cannot reorder candidates. Only apply it when
        # the planner supplied a candidate-specific weave.
        if self.distortion_candidate_specific:
            parts["distortion"] = -round(
                self.distortion * self.distortion_weight
            )
        return parts


@dataclass(frozen=True)
class CandidateScore:
    """Observable components of the existing candidate score."""

    fact: int = 0
    conclusion: int = 0
    grounding: int = 0
    exact_fact: int = 0
    senf: int = 0
    senf_components: Mapping[str, int] = field(default_factory=dict)
    rejected: Optional[str] = None

    @property
    def total(self) -> Optional[int]:
        if self.rejected is not None:
            return None
        value = self.fact + self.conclusion + self.grounding + self.exact_fact + self.senf
        return value if value > 0 else None

    def as_dict(self) -> dict:
        return {
            "fact": self.fact,
            "conclusion": self.conclusion,
            "grounding": self.grounding,
            "exact_fact": self.exact_fact,
            "senf": self.senf,
            "senf_components": dict(self.senf_components),
            "total": self.total,
            "rejected": self.rejected,
        }


def same_shape(left: dict, right: dict) -> bool:
    return left["head"] == right["head"] and left["arity"] == right["arity"]


def signature_can_bind(query: dict, signature: dict) -> bool:
    saw_witness = False
    for q_arg, s_arg in zip(query["args"], signature["args"]):
        if q_arg.startswith(("$", "?")):
            if not s_arg.startswith(("$", "?")):
                saw_witness = True
            continue
        if q_arg != s_arg:
            return False
    return saw_witness or not query["variables"]


def has_witness_path(
    query: dict, matching_facts: list[dict], matching_conclusions: list[dict]
) -> bool:
    if not query["variables"]:
        return True
    for signature in matching_facts + matching_conclusions:
        if signature_can_bind(query, signature):
            return True
    return False


def is_fully_grounded_from_signature(query: dict, matching_facts: list[dict]) -> bool:
    for signature in matching_facts:
        if signature["args"] == query["args"]:
            return True
    return False


def signature_to_query(signature: dict) -> str:
    args = " ".join(signature["args"])
    return f"(: $prf ({signature['head']} {args}) $tv)"


def _ground_with_constants(signature: dict, constants: list[str]) -> Optional[dict]:
    # Signatures are not guaranteed to carry a "variables" key; only parsed
    # queries are. Read variable-ness off the args instead.
    args: list[str] = []
    available = [c for c in constants if c not in signature["args"]]
    for arg in signature["args"]:
        if arg.startswith(("$", "?")):
            if not available:
                return None
            args.append(available.pop(0))
        else:
            args.append(arg)
    return {"head": signature["head"], "args": args, "arity": len(args), "variables": []}


def derive_extra_candidates(
    existing: list[dict],
    facts: list[dict],
    conclusions: list[dict],
    question_constants: list[str],
    limit: int = 4,
) -> list[str]:
    covered = {(sig["head"], sig["arity"]) for sig in existing}
    out: list[str] = []
    for signature in conclusions + facts:
        if len(out) >= limit:
            break
        if signature["arity"] == 0:
            continue
        key = (signature["head"], signature["arity"])
        if key in covered:
            continue
        grounded = _ground_with_constants(signature, question_constants)
        if grounded is None:
            continue
        covered.add(key)
        out.append(signature_to_query(grounded))
    return out


def score_query_candidate(
    query: dict,
    facts: list[dict],
    conclusions: list[dict],
    is_yes_no: bool,
    senf: Optional[SENFSignals] = None,
) -> Optional[int]:
    """Rank a query candidate. Returns None to reject it outright."""
    return score_query_candidate_breakdown(
        query, facts, conclusions, is_yes_no, senf=senf
    ).total


def score_query_candidate_breakdown(
    query: dict,
    facts: list[dict],
    conclusions: list[dict],
    is_yes_no: bool,
    senf: Optional[SENFSignals] = None,
) -> CandidateScore:
    """Return the score components without changing the baseline score API."""
    matching_facts = [sig for sig in facts if same_shape(query, sig)]
    matching_conclusions = [sig for sig in conclusions if same_shape(query, sig)]

    if is_yes_no and query["variables"]:
        if not has_witness_path(query, matching_facts, matching_conclusions):
            return CandidateScore(rejected="no_witness_path")

    senf_components = senf.bonus_breakdown(query) if senf is not None else {}
    return CandidateScore(
        fact=6 if matching_facts else 0,
        conclusion=4 if matching_conclusions else 0,
        grounding=(3 if is_yes_no else 1) if not query["variables"] else (
            3 if not is_yes_no else 0
        ),
        exact_fact=2 if is_fully_grounded_from_signature(query, matching_facts) else 0,
        senf=sum(senf_components.values()),
        senf_components=senf_components,
    )
