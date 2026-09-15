from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable, Optional

from core.statement_validation import (
    is_valid_statement,
    parse_expression,
    render_expression,
)
from core.senf.types import (
    ACTUAL_BRANCH_ID,
    BranchContext,
    Context,
    EntityPersistence,
    SENF,
    ValidityInterval,
)


@dataclass(frozen=True)
class TransportDecision:
    allowed: bool
    reason: str
    source_branch: str
    target_branch: str
    branch_lca: Optional[str] = None
    branch_probability: float = 0.0
    interval_relation: str = "unknown"
    temporal_decay: float = 1.0
    branch_cost: float = 0.0
    temporal_cost: float = 0.0
    persistence_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return round(self.branch_cost + self.temporal_cost + self.persistence_cost, 4)


@dataclass(frozen=True)
class ContextualQuery:
    query: str
    context: Context


class BranchingContextTree:
    def __init__(
        self,
        branches: Iterable[BranchContext],
        max_nodes: int = 64,
        max_depth: int = 16,
    ):
        definitions: dict[str, BranchContext] = {}
        for branch in branches:
            existing = definitions.get(branch.branch_id)
            if existing is not None and existing != branch:
                raise ValueError("conflicting branch definitions")
            definitions[branch.branch_id] = branch
        if len(definitions) > max_nodes:
            raise ValueError("branch limit exceeded")
        if any(
            not math.isfinite(branch.probability)
            or not 0.0 <= branch.probability <= 1.0
            for branch in definitions.values()
        ):
            raise ValueError("invalid branch probability")
        root = definitions.get(ACTUAL_BRANCH_ID)
        if root != BranchContext(ACTUAL_BRANCH_ID, None, "actual", 1.0):
            raise ValueError("invalid actual root")
        if any(
            branch.branch_id != ACTUAL_BRANCH_ID
            and branch.parent_id not in definitions
            for branch in definitions.values()
        ):
            raise ValueError("missing branch parent")
        self._branches = definitions
        for branch_id in sorted(definitions):
            self.lineage(branch_id, max_depth=max_depth)

    @classmethod
    def from_senfs(cls, senfs: Iterable[SENF], **kwargs):
        branches = [branch for senf in senfs for branch in senf.branches]
        if not branches:
            branches = [BranchContext(ACTUAL_BRANCH_ID, None, "actual", 1.0)]
        return cls(branches, **kwargs)

    @property
    def branches(self) -> tuple[BranchContext, ...]:
        return tuple(self._branches[key] for key in sorted(self._branches))

    def get(self, branch_id: str) -> Optional[BranchContext]:
        return self._branches.get(branch_id)

    def lineage(self, branch_id: str, max_depth: int = 16) -> tuple[str, ...]:
        if branch_id not in self._branches:
            raise ValueError("unknown branch")
        path: list[str] = []
        current: Optional[str] = branch_id
        while current is not None:
            if current in path:
                raise ValueError("cyclic branch ancestry")
            path.append(current)
            if len(path) > max_depth:
                raise ValueError("branch depth exceeded")
            current = self._branches[current].parent_id
        if path[-1] != ACTUAL_BRANCH_ID:
            raise ValueError("branch is disconnected from actual root")
        return tuple(reversed(path))

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return ancestor in self.lineage(descendant)

    def least_common_ancestor(self, left: str, right: str) -> str:
        left_path, right_path = self.lineage(left), self.lineage(right)
        lca = ACTUAL_BRANCH_ID
        for left_id, right_id in zip(left_path, right_path):
            if left_id != right_id:
                break
            lca = left_id
        return lca

    def path_probability(self, ancestor: str, descendant: str) -> float:
        lineage = self.lineage(descendant)
        if ancestor not in lineage:
            return 0.0
        start = lineage.index(ancestor) + 1
        probability = 1.0
        for branch_id in lineage[start:]:
            probability *= self._branches[branch_id].probability
        return round(probability, 8)


def _instant(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError("invalid ISO-8601 temporal bound") from exc


def interval_relation(
    left: Optional[ValidityInterval], right: Optional[ValidityInterval]
) -> str:
    if left is None or right is None:
        return "unknown"
    left_start, left_end = _instant(left.start), _instant(left.end)
    right_start, right_end = _instant(right.start), _instant(right.end)
    if left_end is not None and right_start is not None:
        if left_end < right_start or (
            left_end == right_start
            and not (left.end_inclusive and right.start_inclusive)
        ):
            return "before"
    if right_end is not None and left_start is not None:
        if right_end < left_start or (
            right_end == left_start
            and not (right.end_inclusive and left.start_inclusive)
        ):
            return "after"
    if (
        (left_start is None or right_start is not None and left_start <= right_start)
        and (left_end is None or right_end is not None and left_end >= right_end)
    ):
        return "contains"
    if (
        (right_start is None or left_start is not None and right_start <= left_start)
        and (right_end is None or left_end is not None and right_end >= left_end)
    ):
        return "within"
    return "overlaps"


def temporal_decay(
    source: Optional[ValidityInterval], target: Optional[ValidityInterval], rate: float
) -> float:
    relation = interval_relation(source, target)
    if relation not in ("before", "after"):
        return 1.0
    source_point = _instant(source.end if relation == "before" else source.start) if source else None
    target_point = _instant(target.start if relation == "before" else target.end) if target else None
    if source_point is None or target_point is None:
        return 0.0
    gap_days = abs((target_point - source_point).total_seconds()) / 86400.0
    return round(math.exp(-max(0.0, rate) * gap_days), 8)


def assess_transport(
    tree: BranchingContextTree,
    source_context: Context,
    target_context: Context,
    intervals: dict[str, ValidityInterval],
    persistence: Optional[EntityPersistence] = None,
    temporal_decay_rate: float = 0.01,
    purpose: str = "fact",
) -> TransportDecision:
    source_branch = source_context.branch_id
    target_branch = target_context.branch_id
    try:
        lca = tree.least_common_ancestor(source_branch, target_branch)
    except ValueError:
        return TransportDecision(False, "unknown_branch", source_branch, target_branch)

    policy = persistence.persistence_type if persistence else "flexible"
    status = persistence.status if persistence else "realized"
    if status != "realized" and purpose == "fact":
        return TransportDecision(
            False, f"{status}_entity", source_branch, target_branch, lca
        )

    source_is_ancestor = tree.is_ancestor(source_branch, target_branch)
    same_lineage = source_is_ancestor or tree.is_ancestor(target_branch, source_branch)
    if purpose == "fact" and not source_is_ancestor:
        reason = "nonactual_to_actual" if target_branch == ACTUAL_BRANCH_ID else "branch_not_inherited"
        return TransportDecision(False, reason, source_branch, target_branch, lca)
    if purpose == "identity":
        if policy == "rigid":
            branch_allowed = True
        elif policy in ("flexible", "temporal"):
            branch_allowed = same_lineage
        else:
            branch_allowed = source_branch == target_branch
        if not branch_allowed:
            return TransportDecision(
                False, "persistence_branch_conflict", source_branch, target_branch, lca
            )

    source_interval = intervals.get(source_context.validity_interval_id or "")
    target_interval = intervals.get(target_context.validity_interval_id or "")
    relation = interval_relation(source_interval, target_interval)
    decay = temporal_decay(source_interval, target_interval, temporal_decay_rate)
    if policy == "temporal" and relation in ("before", "after"):
        return TransportDecision(
            False, "temporal_identity_expired", source_branch, target_branch, lca,
            interval_relation=relation, temporal_decay=decay,
        )
    probability = tree.path_probability(ACTUAL_BRANCH_ID, target_branch)
    branch_cost = 0.0 if probability >= 1.0 else round(1.0 - probability, 4)
    temporal_cost = round(1.0 - decay, 4)
    persistence_cost = {
        "rigid": 0.0, "flexible": 0.1, "contingent": 0.25, "temporal": 0.2,
    }[policy] if persistence is not None else 0.0
    return TransportDecision(
        True, "allowed", source_branch, target_branch, lca, probability,
        relation, decay, branch_cost, temporal_cost, persistence_cost,
    )


def validate_temporal_model(senf: SENF) -> None:
    BranchingContextTree(senf.branches)
    interval_ids = {interval.interval_id for interval in senf.validity_intervals}
    for interval in senf.validity_intervals:
        start, end = _instant(interval.start), _instant(interval.end)
        if start is not None and end is not None and start > end:
            raise ValueError("reversed validity interval")
    branch_ids = {branch.branch_id for branch in senf.branches}
    entity_ids = {entity.entity_id for entity in senf.entities}
    for item in senf.entity_persistence:
        if item.entity_id not in entity_ids or item.branch_id not in branch_ids:
            raise ValueError("dangling entity persistence")
        if item.validity_interval_id and item.validity_interval_id not in interval_ids:
            raise ValueError("dangling persistence interval")
        if item.status == "unfulfilled" and item.branch_id == ACTUAL_BRANCH_ID:
            raise ValueError("actual entity cannot be unfulfilled")


def unwrap_contextual_query(query: str, source_unit_id: str = "query:u0") -> ContextualQuery:
    """Return the executable inner query and its explicit branch context."""
    try:
        expression = parse_expression(query)
    except ValueError:
        return ContextualQuery(query, Context(source_unit_id))
    if (
        not isinstance(expression, list)
        or len(expression) != 4
        or not isinstance(expression[2], list)
        or len(expression[2]) != 4
        or expression[2][0] != "InContext"
        or not all(isinstance(value, str) and value for value in expression[2][1:3])
    ):
        return ContextualQuery(query, Context(source_unit_id))
    body = expression[2]
    executable = render_expression([expression[0], expression[1], body[3], expression[3]])
    if not executable.startswith("(:"):
        return ContextualQuery(query, Context(source_unit_id))
    return ContextualQuery(
        executable,
        Context(
            source_unit_id,
            branch_id=body[1],
            validity_interval_id=None if body[2] == "none" else body[2],
        ),
    )


def wrap_contextual_statement(statement: str, context: Context) -> str:
    if context.branch_id == ACTUAL_BRANCH_ID and context.validity_interval_id is None:
        return statement
    expression = parse_expression(statement)
    if not isinstance(expression, list) or len(expression) != 4:
        raise ValueError("invalid named statement")
    expression[2] = [
        "InContext",
        context.branch_id,
        context.validity_interval_id or "none",
        expression[2],
    ]
    return render_expression(expression)


def compile_branch_theory(
    senfs: Iterable[SENF],
    target_context: Context,
    max_statements: int = 128,
    temporal_decay_rate: float = 0.01,
    max_branch_nodes: int = 64,
    max_branch_depth: int = 16,
) -> tuple[list[str], tuple[TransportDecision, ...]]:
    """Compile one authorized branch lineage into transient ordinary MeTTa."""
    records = list(senfs)
    tree = BranchingContextTree.from_senfs(
        records, max_nodes=max_branch_nodes, max_depth=max_branch_depth
    )
    intervals: dict[str, ValidityInterval] = {}
    for senf in records:
        for interval in senf.validity_intervals:
            existing = intervals.get(interval.interval_id)
            if existing is not None and existing != interval:
                raise ValueError("conflicting validity interval definitions")
            intervals[interval.interval_id] = interval

    compiled: list[str] = []
    decisions: list[TransportDecision] = []
    seen: set[str] = set()
    persistence_by_symbol: dict[str, list[EntityPersistence]] = {}
    for record in records:
        for item in record.entity_persistence:
            entity = record.entity(item.entity_id)
            if entity is not None:
                persistence_by_symbol.setdefault(entity.canonical_symbol, []).append(item)
    for senf in records:
        for atom in senf.source_atoms:
            try:
                expression = parse_expression(atom)
            except ValueError:
                continue
            if not isinstance(expression, list) or len(expression) != 4:
                continue
            body = expression[2]
            if (
                isinstance(body, list)
                and body
                and body[0] in ("BranchContext", "ValidityInterval", "EntityPersistence")
            ):
                continue
            if (
                isinstance(body, list)
                and len(body) == 4
                and body[0] == "InContext"
            ):
                branch_id, interval_id, content = body[1:]
            else:
                branch_id, interval_id, content = ACTUAL_BRANCH_ID, "none", body
            if not isinstance(branch_id, str) or not isinstance(interval_id, str):
                continue
            source_context = Context(
                target_context.source_unit_id,
                branch_id=branch_id,
                validity_interval_id=None if interval_id == "none" else interval_id,
            )
            content_tokens = set(_expression_tokens(content))
            policies = [
                item
                for symbol in content_tokens
                for item in persistence_by_symbol.get(symbol, ())
                if item.branch_id in (branch_id, target_context.branch_id)
            ]
            atom_decisions = []
            for policy in policies or [None]:
                if (
                    policy is not None
                    and policy.validity_interval_id is not None
                    and policy.validity_interval_id != (
                        target_context.validity_interval_id
                        if policy.branch_id == target_context.branch_id
                        else source_context.validity_interval_id
                    )
                ):
                    atom_decisions.append(TransportDecision(
                        False, "persistence_interval_mismatch", branch_id,
                        target_context.branch_id,
                    ))
                    continue
                atom_decisions.append(assess_transport(
                    tree, source_context, target_context, intervals, policy,
                    temporal_decay_rate=temporal_decay_rate,
                ))
            decisions.extend(atom_decisions)
            if any(not decision.allowed for decision in atom_decisions):
                continue
            truth = expression[3]
            if not (
                isinstance(truth, list)
                and len(truth) == 3
                and truth[0] == "STV"
            ):
                continue
            try:
                strength, weight = float(truth[1]), float(truth[2])
            except (TypeError, ValueError):
                continue
            if (
                not math.isfinite(strength)
                or not math.isfinite(weight)
                or not 0.0 <= strength <= 1.0
                or not 0.0 <= weight <= 1.0
            ):
                continue
            reliability = min(
                decision.branch_probability * decision.temporal_decay
                for decision in atom_decisions
            )
            if reliability <= 0.0 or not math.isfinite(reliability):
                continue
            compiled_atom = render_expression([
                ":",
                f"stage7_{branch_id}_{expression[1]}",
                content,
                ["STV", str(round(strength * reliability, 6)), str(round(weight * reliability, 6))],
            ])
            if compiled_atom in seen or not is_valid_statement(compiled_atom)[0]:
                continue
            seen.add(compiled_atom)
            compiled.append(compiled_atom)
            if len(compiled) >= max(1, max_statements):
                return compiled, tuple(decisions)
    return compiled, tuple(decisions)


def _expression_tokens(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        return ()
    return tuple(token for item in value for token in _expression_tokens(item))
