"""Deterministic, bounded Sinkhorn/IPF scaling implemented with NumPy only."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SinkhornResult:
    """An immutable transport solution and its convergence diagnostics.

    Residuals are maximum absolute marginal errors. ``objective`` is the
    entropy-regularized transport objective for a cost kernel, or
    ``sum(P * (log(P) - log_prior))`` for a log-prior kernel.
    """

    coupling: np.ndarray
    iterations: int
    converged: bool
    row_residual: float
    column_residual: float
    global_residual: float
    objective: float
    entropy: float

    def __post_init__(self) -> None:
        coupling = np.asarray(self.coupling, dtype=np.float64)
        # A bytes-backed view cannot be made writeable again by a caller.
        immutable = np.frombuffer(coupling.tobytes(), dtype=np.float64).reshape(
            coupling.shape
        )
        object.__setattr__(self, "coupling", immutable)


def _vector(value: object, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return result


def _matrix(value: object, name: str, shape: tuple[int, int]) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if result.ndim != 2 or result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis)
    if np.any(~np.isfinite(maximum)):
        raise ValueError("positive marginals have no supported coupling")
    with np.errstate(under="ignore"):
        return maximum + np.log(
            np.sum(np.exp(values - np.expand_dims(maximum, axis)), axis=axis)
        )


def _support_is_feasible(
    support: np.ndarray, rows: np.ndarray, columns: np.ndarray
) -> bool:
    """Check all support cuts with deterministic Dinic max flow."""
    row_count, column_count = support.shape
    source = row_count + column_count
    sink = source + 1
    graph: list[list[list[float | int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: float) -> None:
        forward: list[float | int] = [end, capacity, len(graph[end])]
        reverse: list[float | int] = [start, 0.0, len(graph[start])]
        graph[start].append(forward)
        graph[end].append(reverse)

    total = float(rows.sum())
    scaled_rows = rows / total
    scaled_columns = columns / total
    for row, capacity in enumerate(scaled_rows):
        add_edge(source, row, float(capacity))
    for row, column in np.argwhere(support):
        add_edge(int(row), row_count + int(column), 1.0)
    for column, capacity in enumerate(scaled_columns):
        add_edge(row_count + column, sink, float(capacity))

    flow = 0.0
    epsilon = 32.0 * np.finfo(np.float64).eps
    while True:
        level = [-1] * len(graph)
        level[source] = 0
        queue = [source]
        for node in queue:
            for edge in graph[node]:
                target, capacity, _ = edge
                target = int(target)
                if float(capacity) > epsilon and level[target] < 0:
                    level[target] = level[node] + 1
                    queue.append(target)
        if level[sink] < 0:
            break

        next_edge = [0] * len(graph)

        def push(node: int, amount: float) -> float:
            if node == sink:
                return amount
            while next_edge[node] < len(graph[node]):
                edge = graph[node][next_edge[node]]
                target, capacity, reverse_index = edge
                target = int(target)
                capacity = float(capacity)
                if capacity > epsilon and level[target] == level[node] + 1:
                    sent = push(target, min(amount, capacity))
                    if sent > epsilon:
                        edge[1] = capacity - sent
                        reverse = graph[target][int(reverse_index)]
                        reverse[1] = float(reverse[1]) + sent
                        return sent
                next_edge[node] += 1
            return 0.0

        while True:
            sent = push(source, 1.0 - flow)
            if sent <= epsilon:
                break
            flow += sent
            if flow >= 1.0 - epsilon:
                return True
    return flow >= 1.0 - epsilon


def sinkhorn_ipf(
    row_marginals: object,
    column_marginals: object,
    *,
    cost: Optional[object] = None,
    log_prior: Optional[object] = None,
    support: Optional[object] = None,
    regularization: float = 1.0,
    max_iterations: int = 1_000,
    tolerance: float = 1e-9,
    max_cells: int = 1_000_000,
) -> SinkhornResult:
    """Scale a cost kernel or log-prior to prescribed balanced marginals.

    Exactly one of ``cost`` and ``log_prior`` is required. Structural zeros
    can be supplied through ``support``; ``-inf`` entries in ``log_prior`` are
    also structural zeros. A result is returned after at most
    ``max_iterations`` complete row/column updates.
    """
    rows = _vector(row_marginals, "row_marginals")
    columns = _vector(column_marginals, "column_marginals")
    shape = (rows.size, columns.size)

    if type(max_cells) is not int or max_cells <= 0:
        raise ValueError("max_cells must be a positive integer")
    if rows.size * columns.size > max_cells:
        raise ValueError(f"coupling exceeds max_cells={max_cells}")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    try:
        tolerance = float(tolerance)
        regularization = float(regularization)
    except (TypeError, ValueError) as exc:
        raise ValueError("tolerance and regularization must be numeric") from exc
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be finite and positive")
    if (cost is None) == (log_prior is None):
        raise ValueError("provide exactly one of cost or log_prior")

    with np.errstate(over="ignore"):
        total = float(rows.sum())
        column_total = float(columns.sum())
    if not np.isfinite(total) or not np.isfinite(column_total):
        raise ValueError("marginal total mass must be finite")
    balance_error = abs(total - column_total)
    balance_limit = 32.0 * np.finfo(np.float64).eps * max(
        1.0, total, column_total
    )
    if balance_error > balance_limit:
        raise ValueError("row and column marginals must have equal total mass")

    if support is None:
        allowed = np.ones(shape, dtype=bool)
    else:
        raw_support = np.asarray(support)
        if raw_support.shape != shape or raw_support.ndim != 2:
            raise ValueError(f"support must have shape {shape}")
        if raw_support.dtype.kind != "b":
            raise ValueError("support must be a boolean matrix")
        allowed = raw_support.astype(bool, copy=True)

    cost_matrix: Optional[np.ndarray] = None
    original_log_prior: Optional[np.ndarray] = None
    if cost is not None:
        cost_matrix = _matrix(cost, "cost", shape)
        if not np.all(np.isfinite(cost_matrix)) or np.any(cost_matrix < 0.0):
            raise ValueError("cost must contain finite nonnegative values")
        with np.errstate(over="ignore"):
            log_kernel = -cost_matrix / regularization
        if not np.all(np.isfinite(log_kernel)):
            raise ValueError("cost divided by regularization must remain finite")
    else:
        prior = _matrix(log_prior, "log_prior", shape)
        if np.any(np.isnan(prior)) or np.any(np.isposinf(prior)):
            raise ValueError("log_prior must contain finite values or -inf")
        allowed &= np.isfinite(prior)
        log_kernel = prior.copy()
        original_log_prior = prior.copy()
    log_kernel[~allowed] = -np.inf
    # Scaling is invariant to a global kernel offset. Centering avoids losing
    # small row/column corrections beside very large finite log values.
    finite_kernel = log_kernel[allowed]
    if finite_kernel.size:
        log_kernel[allowed] -= float(np.max(finite_kernel))

    if total == 0.0:
        coupling = np.zeros(shape, dtype=np.float64)
        return SinkhornResult(coupling, 0, True, 0.0, 0.0, 0.0, 0.0, 0.0)

    active_rows = rows > 0.0
    active_columns = columns > 0.0
    active_support = allowed[np.ix_(active_rows, active_columns)]
    active_row_values = rows[active_rows]
    active_column_values = columns[active_columns]
    if not _support_is_feasible(
        active_support, active_row_values, active_column_values
    ):
        raise ValueError("marginals are impossible on the supplied support")

    kernel = log_kernel[np.ix_(active_rows, active_columns)]
    log_rows = np.log(active_row_values)
    log_columns = np.log(active_column_values)
    row_scale = np.zeros_like(log_rows)
    column_scale = np.zeros_like(log_columns)
    converged = False
    active_coupling = np.zeros_like(kernel)
    row_residual = column_residual = global_residual = float("inf")

    for iterations in range(1, max_iterations + 1):
        row_scale = log_rows - _logsumexp(kernel + column_scale[None, :], 1)
        column_scale = log_columns - _logsumexp(
            kernel + row_scale[:, None], 0
        )
        try:
            with np.errstate(under="ignore", over="raise", invalid="raise"):
                active_coupling = np.exp(
                    kernel + row_scale[:, None] + column_scale[None, :]
                )
        except FloatingPointError as exc:
            raise ValueError("non-finite coupling produced during scaling") from exc
        if not np.all(np.isfinite(active_coupling)):
            raise ValueError("non-finite coupling produced during scaling")
        row_residual = float(
            np.max(np.abs(active_coupling.sum(axis=1) - active_row_values))
        )
        column_residual = float(
            np.max(np.abs(active_coupling.sum(axis=0) - active_column_values))
        )
        mass_residual = abs(float(active_coupling.sum()) - total)
        global_residual = max(row_residual, column_residual, mass_residual)
        if global_residual <= tolerance:
            converged = True
            break

    coupling = np.zeros(shape, dtype=np.float64)
    coupling[np.ix_(active_rows, active_columns)] = active_coupling
    positive = coupling > 0.0
    entropy = -float(np.sum(coupling[positive] * np.log(coupling[positive])))
    if cost_matrix is not None:
        objective = float(np.sum(coupling * cost_matrix) - regularization * entropy)
    else:
        assert original_log_prior is not None
        objective = float(
            np.sum(coupling[positive] * (
                np.log(coupling[positive]) - original_log_prior[positive]
            ))
        )
    if not np.isfinite(objective) or not np.isfinite(entropy):
        raise ValueError("non-finite solution diagnostics")
    return SinkhornResult(
        coupling,
        iterations,
        converged,
        row_residual,
        column_residual,
        global_residual,
        objective,
        entropy,
    )


# Short names keep the standalone module convenient without package exports.
solve_sinkhorn = sinkhorn_ipf
sinkhorn = sinkhorn_ipf


__all__ = ["SinkhornResult", "sinkhorn", "sinkhorn_ipf", "solve_sinkhorn"]
