from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from core.senf.sinkhorn import SinkhornResult, sinkhorn_ipf


def test_cost_solver_matches_marginals_deterministically():
    kwargs = dict(
        cost=[[0.0, 2.0], [1.0, 0.0]],
        regularization=0.4,
        tolerance=1e-12,
    )
    first = sinkhorn_ipf([0.4, 0.6], [0.5, 0.5], **kwargs)
    second = sinkhorn_ipf([0.4, 0.6], [0.5, 0.5], **kwargs)

    assert first.converged
    assert first.iterations == second.iterations
    np.testing.assert_array_equal(first.coupling, second.coupling)
    np.testing.assert_allclose(first.coupling.sum(axis=1), [0.4, 0.6], atol=1e-12)
    np.testing.assert_allclose(first.coupling.sum(axis=0), [0.5, 0.5], atol=1e-12)
    assert first.global_residual <= 1e-12
    assert np.isfinite(first.objective)
    assert first.entropy > 0.0


def test_log_prior_and_support_preserve_structural_zeros():
    result = sinkhorn_ipf(
        [0.5, 0.5],
        [0.25, 0.75],
        log_prior=[[0.0, 0.0], [-np.inf, 0.0]],
        support=np.array([[True, True], [True, True]]),
        tolerance=1e-12,
    )

    assert result.converged
    assert result.coupling[1, 0] == 0.0
    np.testing.assert_allclose(result.coupling.sum(axis=1), [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(result.coupling.sum(axis=0), [0.25, 0.75], atol=1e-12)


def test_large_log_prior_offset_is_numerically_invariant():
    result = sinkhorn_ipf(
        [0.5, 0.5],
        [0.5, 0.5],
        log_prior=np.full((2, 2), 1e300),
    )

    assert result.converged
    np.testing.assert_allclose(result.coupling, np.full((2, 2), 0.25))


def test_zero_marginals_have_zero_rows_and_columns():
    result = sinkhorn_ipf(
        [0.0, 1.0], [1.0, 0.0], cost=np.zeros((2, 2)), tolerance=1e-12
    )

    np.testing.assert_array_equal(result.coupling, [[0.0, 0.0], [1.0, 0.0]])
    assert result.converged


def test_solution_and_coupling_are_immutable():
    result = sinkhorn_ipf([1.0], [1.0], cost=[[0.0]])

    with pytest.raises(FrozenInstanceError):
        result.iterations = 3
    with pytest.raises(ValueError):
        result.coupling[0, 0] = 0.0
    with pytest.raises(ValueError):
        result.coupling.setflags(write=True)
    assert isinstance(result, SinkhornResult)


def test_iteration_bound_returns_diagnostics_without_claiming_convergence():
    result = sinkhorn_ipf(
        [0.9, 0.1],
        [0.1, 0.9],
        cost=[[0.0, 20.0], [20.0, 0.0]],
        regularization=0.1,
        max_iterations=1,
        tolerance=1e-15,
    )

    assert result.iterations == 1
    assert not result.converged
    assert result.global_residual > 1e-15
    assert np.all(np.isfinite(result.coupling))


def test_hall_infeasible_support_fails_closed():
    support = np.array([
        [True, False, False],
        [True, False, False],
        [False, True, True],
    ])

    with pytest.raises(ValueError, match="impossible"):
        sinkhorn_ipf(
            [0.4, 0.4, 0.2],
            [0.5, 0.25, 0.25],
            cost=np.zeros((3, 3)),
            support=support,
        )


@pytest.mark.parametrize(
    ("rows", "columns", "cost"),
    [
        ([np.nan], [1.0], [[0.0]]),
        ([-1.0], [-1.0], [[0.0]]),
        ([1.0], [1.0], [[np.inf]]),
        ([1.0], [1.0], [[-1.0]]),
        ([1.0], [2.0], [[0.0]]),
    ],
)
def test_invalid_numeric_inputs_fail_closed(rows, columns, cost):
    with pytest.raises(ValueError):
        sinkhorn_ipf(rows, columns, cost=cost)


def test_rejects_nan_log_prior_and_cell_limit_before_allocation():
    with pytest.raises(ValueError, match="log_prior"):
        sinkhorn_ipf([1.0], [1.0], log_prior=[[np.nan]])
    with pytest.raises(ValueError, match="max_cells"):
        sinkhorn_ipf([0.5, 0.5], [0.5, 0.5], cost=np.zeros((2, 2)), max_cells=3)


def test_all_zero_balanced_problem_is_well_defined():
    result = sinkhorn_ipf([0.0, 0.0], [0.0], cost=[[1.0], [2.0]])

    assert result.iterations == 0
    assert result.converged
    np.testing.assert_array_equal(result.coupling, np.zeros((2, 1)))
    assert result.objective == result.entropy == result.global_residual == 0.0
