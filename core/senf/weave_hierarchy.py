"""Bounded hierarchical Sinkhorn polish and deterministic hard rounding."""

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

import numpy as np

from core.senf.sinkhorn import SinkhornResult, sinkhorn_ipf
from core.senf.weave_model import (
    FrameAlignment,
    PolishDiagnostics,
    PolishStageDiagnostics,
    SourceFrameKey,
)


TransportKey = tuple[str, SourceFrameKey | None]


class HierarchyFallback(RuntimeError):
    """A closed failure that must return a safe unaligned result."""


@dataclass(frozen=True)
class HierarchyLimits:
    global_candidate_cap: int
    max_cells: int
    max_seeds: int
    max_iterations: int
    tolerance: float
    regularization: float
    forget_fine_costs: bool = False


@dataclass(frozen=True)
class HierarchyResult:
    selections: tuple[tuple[FrameAlignment, ...], ...]
    diagnostics: PolishDiagnostics


def _alignment_key(item: FrameAlignment) -> tuple:
    return (
        item.query_frame_id,
        item.source_frame.source_id,
        item.source_frame.frame_id,
        round(item.costs.total, 12),
        -item.score,
        item.evidence,
    )


def _stage(solutions: Sequence[SinkhornResult]) -> PolishStageDiagnostics:
    return PolishStageDiagnostics(
        solves=len(solutions),
        iterations=sum(item.iterations for item in solutions),
        converged=all(item.converged for item in solutions),
        row_residual=max((item.row_residual for item in solutions), default=0.0),
        column_residual=max(
            (item.column_residual for item in solutions), default=0.0
        ),
        global_residual=max(
            (item.global_residual for item in solutions), default=0.0
        ),
        objective=sum(item.objective for item in solutions),
        entropy=sum(item.entropy for item in solutions),
    )


def _components(
    candidates: Sequence[FrameAlignment],
) -> tuple[tuple[FrameAlignment, ...], ...]:
    by_query: dict[str, list[FrameAlignment]] = {}
    by_source: dict[SourceFrameKey, list[FrameAlignment]] = {}
    for item in candidates:
        by_query.setdefault(item.query_frame_id, []).append(item)
        by_source.setdefault(item.source_frame, []).append(item)
    unseen = set(candidates)
    result = []
    while unseen:
        first = min(unseen, key=_alignment_key)
        pending = [first]
        component: set[FrameAlignment] = set()
        while pending:
            item = pending.pop()
            if item in component:
                continue
            component.add(item)
            pending.extend(
                edge
                for edge in by_query[item.query_frame_id]
                + by_source[item.source_frame]
                if edge not in component
            )
        unseen.difference_update(component)
        result.append(tuple(sorted(component, key=_alignment_key)))
    return tuple(result)


def _solve(
    candidates: Sequence[FrameAlignment],
    limits: HierarchyLimits,
    prior: dict[TransportKey, float] | None = None,
) -> tuple[SinkhornResult, dict[TransportKey, float]]:
    queries = sorted({item.query_frame_id for item in candidates})
    sources = sorted({item.source_frame for item in candidates})
    query_index = {value: index for index, value in enumerate(queries)}
    source_index = {value: index for index, value in enumerate(sources)}
    row_count = len(queries) + len(sources)
    column_count = len(sources) + len(queries)
    if row_count * column_count > limits.max_cells:
        raise HierarchyFallback("sinkhorn_max_cells")

    support = np.zeros((row_count, column_count), dtype=bool)
    log_prior = np.full((row_count, column_count), -np.inf, dtype=np.float64)
    for item in candidates:
        row = query_index[item.query_frame_id]
        column = source_index[item.source_frame]
        key = (item.query_frame_id, item.source_frame)
        inherited = 1.0 if prior is None else max(prior.get(key, 0.0), 1e-300)
        support[row, column] = True
        log_prior[row, column] = (
            -(0.0 if limits.forget_fine_costs else item.costs.total)
            / limits.regularization
            if prior is None else math.log(inherited)
        )

    # Every real query may remain unmatched. Source-slack rows absorb unused
    # source capacity and balance the unmatched columns.
    for row in range(len(queries)):
        column = len(sources) + row
        support[row, column] = True
        key = (queries[row], None)
        inherited = (
            1.0 if prior is None
            else max(prior.get(key, math.exp(-1.0 / limits.regularization)), 1e-300)
        )
        log_prior[row, column] = (
            -1.0 / limits.regularization
            if prior is None else math.log(inherited)
        )
    for source_offset in range(len(sources)):
        row = len(queries) + source_offset
        support[row, source_offset] = True
        log_prior[row, source_offset] = 0.0
        support[row, len(sources):] = True
        log_prior[row, len(sources):] = 0.0

    try:
        solved = sinkhorn_ipf(
            np.ones(row_count),
            np.ones(column_count),
            log_prior=log_prior,
            support=support,
            max_iterations=limits.max_iterations,
            tolerance=limits.tolerance,
            max_cells=limits.max_cells,
        )
    except ValueError as exc:
        raise HierarchyFallback(f"sinkhorn_invalid:{exc}") from exc
    if not solved.converged:
        raise HierarchyFallback("sinkhorn_not_converged")
    weights = {
        (item.query_frame_id, item.source_frame): float(
            solved.coupling[
                query_index[item.query_frame_id], source_index[item.source_frame]
            ]
        )
        for item in candidates
    }
    weights.update({
        (query_id, None): float(
            solved.coupling[query_index[query_id], len(sources) + query_index[query_id]]
        )
        for query_id in queries
    })
    return solved, weights


def _assignment_key(
    selected: Sequence[FrameAlignment],
    query_frame_ids: Sequence[str],
) -> tuple[tuple[str, str, str], ...]:
    by_query = {item.query_frame_id: item.source_frame for item in selected}
    result = []
    for query_id in sorted(set(query_frame_ids)):
        source = by_query.get(query_id)
        result.append((
            query_id,
            source.source_id if source is not None else "",
            source.frame_id if source is not None else "",
        ))
    return tuple(result)


def _maximum_assignment(
    candidates: Sequence[FrameAlignment],
    weights: dict[TransportKey, float],
    query_frame_ids: Sequence[str],
    max_cells: int,
    excluded: frozenset[TransportKey] = frozenset(),
) -> tuple[tuple[FrameAlignment, ...], tuple[TransportKey, ...], float] | None:
    """Solve the exact maximum-weight matching with deterministic Hungarian."""
    query_ids = sorted(set(query_frame_ids))
    if not query_ids:
        return (), (), 0.0
    sources = sorted({item.source_frame for item in candidates})
    source_index = {source: index for index, source in enumerate(sources)}
    candidate_by_edge = {
        (item.query_frame_id, item.source_frame): item for item in candidates
    }
    column_count = len(sources) + len(query_ids)
    if len(query_ids) * column_count > max_cells:
        raise HierarchyFallback("assignment_max_cells")
    forbidden = 1e100
    costs = [[forbidden] * column_count for _ in query_ids]
    for row, query_id in enumerate(query_ids):
        unmatched = (query_id, None)
        if unmatched not in excluded:
            costs[row][len(sources) + row] = -weights.get(unmatched, 0.0)
        for item in candidates:
            if item.query_frame_id != query_id:
                continue
            edge = (query_id, item.source_frame)
            if edge not in excluded:
                costs[row][source_index[item.source_frame]] = -weights[edge]

    # Rectangular Hungarian algorithm, with one row per query and at least one
    # private unmatched column per row. Strict comparisons make ties stable.
    row_count = len(query_ids)
    u = [0.0] * (row_count + 1)
    v = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    path = [0] * (column_count + 1)
    for row in range(1, row_count + 1):
        matched_row[0] = row
        minimum = [forbidden] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            active_row = matched_row[column]
            delta = forbidden
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                reduced = (
                    costs[active_row - 1][candidate_column - 1]
                    - u[active_row]
                    - v[candidate_column]
                )
                if reduced < minimum[candidate_column]:
                    minimum[candidate_column] = reduced
                    path[candidate_column] = column
                if minimum[candidate_column] < delta:
                    delta = minimum[candidate_column]
                    next_column = candidate_column
            if delta >= forbidden / 2:
                return None
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    u[matched_row[candidate_column]] += delta
                    v[candidate_column] -= delta
                else:
                    minimum[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            previous = path[column]
            matched_row[column] = matched_row[previous]
            column = previous
            if column == 0:
                break

    assigned_columns = [0] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column]:
            assigned_columns[matched_row[column] - 1] = column - 1
    selected = []
    assignment = []
    mass = 0.0
    for row, column in enumerate(assigned_columns):
        query_id = query_ids[row]
        if costs[row][column] >= forbidden / 2:
            return None
        source = sources[column] if column < len(sources) else None
        edge = (query_id, source)
        if edge in excluded or source is None and column != len(sources) + row:
            return None
        assignment.append(edge)
        mass += weights.get(edge, 0.0)
        if source is not None:
            selected.append(candidate_by_edge[edge])
    selected.sort(key=_alignment_key)
    return tuple(selected), tuple(assignment), mass


def _round(
    candidates: Sequence[FrameAlignment],
    weights: dict[TransportKey, float],
    query_frame_ids: Sequence[str],
    max_seeds: int,
    max_cells: int,
) -> tuple[tuple[FrameAlignment, ...], ...]:
    """Return exact coupling-optimal assignments and bounded exclusions.

    Sinkhorn mass is a polish objective. Callers rerank these exact alternatives
    with the final semantic objective, including unmatched and consistency cost.
    """
    target = max(0, max_seeds - 1)
    if target == 0:
        return ((),)
    initial = _maximum_assignment(
        candidates, weights, query_frame_ids, max_cells
    )
    if initial is None:
        return ((),)
    frontier: list[
        tuple[float, tuple, tuple, tuple, frozenset[TransportKey]]
    ] = []
    queued: set[frozenset[TransportKey]] = {frozenset()}

    def enqueue(
        exclusions: frozenset[TransportKey],
        solved: tuple[tuple[FrameAlignment, ...], tuple[TransportKey, ...], float],
    ) -> None:
        selected, assignment, mass = solved
        assignment_key = tuple(
            (
                query_id,
                source.source_id if source is not None else "",
                source.frame_id if source is not None else "",
            )
            for query_id, source in assignment
        )
        exclusion_key = tuple(sorted(
            (
                query_id,
                source.source_id if source is not None else "",
                source.frame_id if source is not None else "",
            )
            for query_id, source in exclusions
        ))
        heapq.heappush(frontier, (
            -mass,
            _assignment_key(selected, query_frame_ids),
            assignment_key,
            exclusion_key,
            exclusions,
        ))

    solutions = {frozenset(): initial}
    enqueue(frozenset(), initial)
    unique: dict[tuple, tuple[FrameAlignment, ...]] = {}
    solve_count = 1
    max_solves = max_seeds * max(1, len(set(query_frame_ids)))
    while frontier and len(unique) < target:
        _, _, _, _, exclusions = heapq.heappop(frontier)
        selected, assignment, _ = solutions.pop(exclusions)
        if selected:
            key = tuple(_alignment_key(item) for item in selected)
            unique.setdefault(key, selected)
        for edge in assignment:
            if solve_count >= max_solves:
                break
            child = exclusions | {edge}
            if child in queued:
                continue
            queued.add(child)
            solve_count += 1
            solved = _maximum_assignment(
                candidates, weights, query_frame_ids, max_cells, child
            )
            if solved is not None:
                solutions[child] = solved
                enqueue(child, solved)
    return tuple(unique.values())[:target] + ((),)


def polish_hierarchy(
    candidates: Sequence[FrameAlignment],
    query_frame_ids: Sequence[str],
    limits: HierarchyLimits,
) -> HierarchyResult:
    integer_limits = (
        limits.global_candidate_cap,
        limits.max_cells,
        limits.max_seeds,
        limits.max_iterations,
    )
    if any(type(value) is not int or value <= 0 for value in integer_limits):
        raise HierarchyFallback("invalid_hierarchy_bound")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in (limits.tolerance, limits.regularization)
    ):
        raise HierarchyFallback("invalid_hierarchy_numeric_setting")
    if not candidates:
        raise HierarchyFallback("no_hard_compatible_candidates")
    if len(candidates) > limits.global_candidate_cap:
        raise HierarchyFallback("global_candidate_cap")

    ordered = tuple(sorted(candidates, key=_alignment_key))
    local_solutions = []
    local_weights: dict[TransportKey, float] = {}
    for component in _components(ordered):
        solved, weights = _solve(component, limits)
        local_solutions.append(solved)
        local_weights.update(weights)

    block_solutions = []
    block_weights = dict(local_weights)
    for source_id in sorted({item.source_frame.source_id for item in ordered}):
        source_items = tuple(
            item for item in ordered if item.source_frame.source_id == source_id
        )
        source_frames = sorted({item.source_frame for item in source_items})
        width = 1
        while True:
            next_weights = dict(block_weights)
            for start in range(0, len(source_frames), width):
                frame_block = set(source_frames[start:start + width])
                block = tuple(
                    item for item in source_items if item.source_frame in frame_block
                )
                solved, weights = _solve(block, limits, block_weights)
                block_solutions.append(solved)
                next_weights.update(
                    (key, value) for key, value in weights.items()
                    if key[1] is not None
                )
            block_weights = next_weights
            if width >= len(source_frames):
                break
            width = min(width * 2, len(source_frames))

    global_solution, global_weights = _solve(ordered, limits, block_weights)
    selections = _round(
        ordered, global_weights, query_frame_ids, limits.max_seeds, limits.max_cells
    )
    diagnostics = PolishDiagnostics(
        engine="hierarchical",
        local=_stage(local_solutions),
        block=_stage(block_solutions),
        global_stage=_stage((global_solution,)),
        candidate_count=len(ordered),
        seed_count=len(selections),
        matched_soft_mass=sum(
            value for (query_id, source), value in global_weights.items()
            if source is not None
        ),
        unmatched_soft_mass=sum(
            value for (query_id, source), value in global_weights.items()
            if source is None
        ),
    )
    return HierarchyResult(selections, diagnostics)


__all__ = [
    "HierarchyFallback",
    "HierarchyLimits",
    "HierarchyResult",
    "polish_hierarchy",
]
