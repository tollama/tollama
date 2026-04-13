"""Hierarchical forecast reconciliation.

Enforces coherency for hierarchical time series (e.g., total = sum of regions)
using standard reconciliation methods: bottom-up, top-down, OLS, and MinT shrink.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .schemas import ForecastResponse, SeriesForecast

ReconciliationMethod = Literal["bottom_up", "top_down", "ols", "mint_shrink"]

logger = logging.getLogger(__name__)


class ReconciliationError(ValueError):
    """Raised when reconciliation inputs are invalid or inconsistent."""


@dataclass
class HierarchySpec:
    """Specification of a hierarchical time-series structure.

    Args:
        tree: Mapping from each parent node to its direct children.
            Example: ``{"total": ["region_a", "region_b"]}``.
    """

    tree: dict[str, list[str]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def _all_parents(self) -> set[str]:
        """Return the set of all parent (non-leaf) node IDs."""
        return set(self.tree.keys())

    def _all_children(self) -> set[str]:
        """Return the union of all child lists."""
        children: set[str] = set()
        for child_list in self.tree.values():
            children.update(child_list)
        return children

    def leaves(self) -> list[str]:
        """Return bottom-level (leaf) series IDs in deterministic order."""
        parents = self._all_parents()
        all_children = self._all_children()
        leaf_set = all_children - parents
        # Also include parents that have no parent themselves but are not in
        # any child list (single-node trees); however the common case is that
        # roots are parents only.
        # Deterministic order: BFS from roots, keeping only leaves.
        ordered: list[str] = []
        visited: set[str] = set()
        queue: deque[str] = deque(self._roots())
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            if node in leaf_set:
                ordered.append(node)
            if node in self.tree:
                for child in self.tree[node]:
                    queue.append(child)
        return ordered

    def _roots(self) -> list[str]:
        """Return root nodes (parents that are not children of any other node)."""
        all_children = self._all_children()
        return [p for p in self.tree if p not in all_children]

    def all_series_ids(self) -> list[str]:
        """Return all series IDs ordered top-down (BFS), parents before children."""
        ordered: list[str] = []
        visited: set[str] = set()
        queue: deque[str] = deque(self._roots())
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            ordered.append(node)
            if node in self.tree:
                for child in self.tree[node]:
                    queue.append(child)
        return ordered

    def validate_series_ids(self, series_ids: set[str]) -> list[str]:
        """Check whether *series_ids* covers the hierarchy.

        Returns:
            A list of missing series ID descriptions (empty if fully covered).
        """
        expected = set(self.all_series_ids())
        missing = expected - series_ids
        if not missing:
            return []
        return [f"missing series in hierarchy: {sorted(missing)}"]

    def summing_matrix(self) -> np.ndarray:
        """Build the summing matrix **S**.

        ``S`` has shape ``(n_all, n_bottom)`` where
        ``S @ bottom_forecasts == all_level_forecasts``.
        Rows correspond to :meth:`all_series_ids` order; columns correspond
        to :meth:`leaves` order.
        """
        return build_summing_matrix(self)


# ------------------------------------------------------------------
# Summing matrix construction
# ------------------------------------------------------------------


def build_summing_matrix(spec: HierarchySpec) -> np.ndarray:
    """Build the summing matrix **S** for a hierarchy.

    The matrix has shape ``(n_all, n_bottom)`` such that
    ``S @ bottom_forecasts`` produces the full vector of coherent forecasts
    for every level in the hierarchy.

    Args:
        spec: Hierarchy specification.

    Returns:
        Numpy array of shape ``(n_all, n_bottom)``.
    """
    all_ids = spec.all_series_ids()
    leaf_ids = spec.leaves()

    if not leaf_ids:
        raise ReconciliationError("hierarchy has no leaf nodes")

    n_all = len(all_ids)
    n_bottom = len(leaf_ids)

    id_to_row: dict[str, int] = {sid: i for i, sid in enumerate(all_ids)}
    leaf_to_col: dict[str, int] = {sid: i for i, sid in enumerate(leaf_ids)}

    s_matrix = np.zeros((n_all, n_bottom), dtype=np.float64)

    # Leaf rows: identity mapping
    for leaf_id, col in leaf_to_col.items():
        row = id_to_row[leaf_id]
        s_matrix[row, col] = 1.0

    # Aggregate rows: sum of descendant leaves (bottom-up from leaves).
    # Process in reverse BFS order so children are resolved before parents.
    for node_id in reversed(all_ids):
        if node_id in leaf_to_col:
            continue  # already set
        row = id_to_row[node_id]
        children = spec.tree.get(node_id, [])
        for child_id in children:
            child_row = id_to_row[child_id]
            s_matrix[row, :] += s_matrix[child_row, :]

    return s_matrix


# ------------------------------------------------------------------
# Reconciliation
# ------------------------------------------------------------------


def reconcile(
    response: ForecastResponse,
    spec: HierarchySpec,
    method: ReconciliationMethod = "bottom_up",
    proportions: dict[str, float] | None = None,
) -> ForecastResponse:
    """Reconcile a :class:`ForecastResponse` to satisfy hierarchy constraints.

    Args:
        response: Original (possibly incoherent) forecast response.
        spec: Hierarchy specification.
        method: Reconciliation approach.
        proportions: For ``top_down`` only -- mapping from leaf series ID to its
            share of the parent total.  Values should sum to 1 within each
            parent's children.  If *None* and method is ``top_down``, proportions
            are estimated from the base forecasts (average share).
    Returns:
        A new :class:`ForecastResponse` with reconciled ``mean`` values.
    """
    all_ids = spec.all_series_ids()
    leaf_ids = spec.leaves()
    s_matrix = build_summing_matrix(spec)

    # Map series IDs to their forecast objects.
    fc_map: dict[str, SeriesForecast] = {fc.id: fc for fc in response.forecasts}

    # Identify series outside the hierarchy -- they will be passed through.
    hierarchy_set = set(all_ids)
    outside_ids = [fc.id for fc in response.forecasts if fc.id not in hierarchy_set]

    # Validate that we have at least the bottom-level series.
    missing_leaves = [lid for lid in leaf_ids if lid not in fc_map]
    if method == "bottom_up" and missing_leaves:
        raise ReconciliationError(f"bottom_up requires all leaf series; missing: {missing_leaves}")

    # Determine horizon from first available hierarchy series.
    horizon: int | None = None
    for sid in all_ids:
        if sid in fc_map:
            horizon = len(fc_map[sid].mean)
            break
    if horizon is None:
        raise ReconciliationError("no hierarchy series found in forecast response")

    # Assemble base forecast matrix y_hat: shape (n_all, horizon).
    n_all = len(all_ids)
    y_hat = np.zeros((n_all, horizon), dtype=np.float64)
    for i, sid in enumerate(all_ids):
        if sid in fc_map:
            vals = fc_map[sid].mean
            if len(vals) != horizon:
                raise ReconciliationError(
                    f"series {sid!r} horizon {len(vals)} != expected {horizon}"
                )
            y_hat[i, :] = [float(v) for v in vals]

    # Bottom-level forecast matrix: shape (n_bottom, horizon).
    leaf_indices = [all_ids.index(lid) for lid in leaf_ids]
    y_bottom = y_hat[leaf_indices, :]

    warnings: list[str] = list(response.warnings or [])

    if method == "bottom_up":
        reconciled = s_matrix @ y_bottom
        warnings.append("reconciliation: bottom_up applied")

    elif method == "top_down":
        reconciled = _top_down(
            y_hat,
            s_matrix,
            spec,
            all_ids,
            leaf_ids,
            proportions,
        )
        warnings.append("reconciliation: top_down applied")

    elif method == "ols":
        reconciled = _ols(y_hat, s_matrix)
        warnings.append("reconciliation: ols applied")

    elif method == "mint_shrink":
        reconciled = _mint_shrink(y_hat, s_matrix)
        warnings.append("reconciliation: mint_shrink applied")

    else:  # pragma: no cover
        raise ReconciliationError(f"unknown method: {method!r}")

    if outside_ids:
        warnings.append(
            f"reconciliation: {len(outside_ids)} series outside hierarchy passed through unchanged"
        )

    # Build new SeriesForecast objects.
    id_to_row: dict[str, int] = {sid: i for i, sid in enumerate(all_ids)}
    new_forecasts: list[SeriesForecast] = []

    for sid in all_ids:
        row = id_to_row[sid]
        base = fc_map.get(sid)
        new_mean = reconciled[row, :].tolist()
        if base is not None:
            new_forecasts.append(
                SeriesForecast(
                    id=base.id,
                    freq=base.freq,
                    start_timestamp=base.start_timestamp,
                    mean=new_mean,
                    quantiles=base.quantiles,
                )
            )
        else:
            # Series exists in hierarchy but not in original response -- use a
            # representative series for metadata.
            ref = next(fc for fc in response.forecasts if fc.id in hierarchy_set)
            new_forecasts.append(
                SeriesForecast(
                    id=sid,
                    freq=ref.freq,
                    start_timestamp=ref.start_timestamp,
                    mean=new_mean,
                    quantiles=None,
                )
            )

    # Append pass-through series.
    for sid in outside_ids:
        new_forecasts.append(fc_map[sid])

    return ForecastResponse(
        model=response.model,
        forecasts=new_forecasts,
        metrics=response.metrics,
        timing=response.timing,
        explanation=response.explanation,
        narrative=response.narrative,
        usage=response.usage,
        warnings=warnings or None,
    )


# ------------------------------------------------------------------
# Method implementations
# ------------------------------------------------------------------


def _top_down(
    y_hat: np.ndarray,
    s_matrix: np.ndarray,
    spec: HierarchySpec,
    all_ids: list[str],
    leaf_ids: list[str],
    proportions: dict[str, float] | None,
) -> np.ndarray:
    """Top-down reconciliation: disaggregate from root to leaves."""
    n_bottom = len(leaf_ids)
    horizon = y_hat.shape[1]
    id_to_row = {sid: i for i, sid in enumerate(all_ids)}
    leaf_to_col = {sid: i for i, sid in enumerate(leaf_ids)}

    if proportions is None:
        # Estimate proportions from base forecasts: average share over horizon.
        proportions = _estimate_proportions(y_hat, spec, all_ids, leaf_ids)

    # Validate proportions cover all leaves.
    missing = [lid for lid in leaf_ids if lid not in proportions]
    if missing:
        raise ReconciliationError(f"top_down proportions missing for leaves: {missing}")

    # Disaggregate each root's forecast to its descendant leaves.
    bottom_reconciled = np.zeros((n_bottom, horizon), dtype=np.float64)

    for root_id in spec._roots():
        root_row = id_to_row[root_id]
        root_forecast = y_hat[root_row, :]
        # Find all leaves that are descendants of this root.
        root_leaves = [lid for lid in leaf_ids if s_matrix[root_row, leaf_to_col[lid]] > 0]
        for lid in root_leaves:
            col = leaf_to_col[lid]
            bottom_reconciled[col, :] = root_forecast * proportions[lid]

    return s_matrix @ bottom_reconciled


def _estimate_proportions(
    y_hat: np.ndarray,
    spec: HierarchySpec,
    all_ids: list[str],
    leaf_ids: list[str],
) -> dict[str, float]:
    """Estimate top-down proportions from base forecasts."""
    id_to_row = {sid: i for i, sid in enumerate(all_ids)}
    proportions: dict[str, float] = {}

    for parent_id, children in spec.tree.items():
        parent_row = id_to_row[parent_id]
        parent_mean = float(np.mean(y_hat[parent_row, :]))
        if abs(parent_mean) < 1e-12:
            # Uniform split when parent is near zero.
            leaf_children = [c for c in children if c not in spec.tree]
            n = max(len(leaf_children), 1)
            for cid in leaf_children:
                proportions[cid] = 1.0 / n
            continue

        for cid in children:
            if cid not in spec.tree:
                # Direct leaf child.
                child_row = id_to_row.get(cid)
                if child_row is not None:
                    child_mean = float(np.mean(y_hat[child_row, :]))
                    proportions[cid] = child_mean / parent_mean
                else:
                    proportions[cid] = 0.0

    return proportions


def _ols(y_hat: np.ndarray, s_matrix: np.ndarray) -> np.ndarray:
    """OLS (ordinary least squares) reconciliation.

    Projection matrix: ``P = (S'S)^{-1} S'``, reconciled = ``S @ P @ y_hat``.
    """
    sts = s_matrix.T @ s_matrix
    try:
        sts_inv = np.linalg.inv(sts)
    except np.linalg.LinAlgError as exc:
        raise ReconciliationError(
            "S'S is singular; OLS reconciliation not possible for this hierarchy"
        ) from exc
    p_matrix = sts_inv @ s_matrix.T
    return s_matrix @ p_matrix @ y_hat


def _mint_shrink(y_hat: np.ndarray, s_matrix: np.ndarray) -> np.ndarray:
    """MinT shrinkage reconciliation.

    Uses the Ledoit-Wolf-style shrinkage covariance estimator on the
    in-sample residuals (approximated from the base forecasts).

    Projection: ``P = (S' W^{-1} S)^{-1} S' W^{-1}``,
    reconciled = ``S @ P @ y_hat``.
    """
    n_all, horizon = y_hat.shape

    # Estimate residuals as deviation from S @ bottom_hat.
    n_bottom = s_matrix.shape[1]
    # Extract bottom rows: they correspond to rows where S has a single 1.
    bottom_mask = np.array([np.count_nonzero(s_matrix[i, :]) == 1 for i in range(n_all)])
    bottom_rows = np.where(bottom_mask)[0]
    if len(bottom_rows) != n_bottom:
        # Fallback: just pick the last n_bottom rows.
        bottom_rows = np.arange(n_all - n_bottom, n_all)

    y_bottom = y_hat[bottom_rows, :]
    coherent_hat = s_matrix @ y_bottom
    residuals = y_hat - coherent_hat  # (n_all, horizon)

    # Shrinkage covariance estimator (Ledoit-Wolf target: diagonal).
    if horizon < 2:
        # Not enough data for covariance estimation; fall back to OLS.
        logger.warning("mint_shrink: horizon < 2, falling back to OLS reconciliation")
        return _ols(y_hat, s_matrix)

    sample_cov = np.cov(residuals, rowvar=True)  # (n_all, n_all)
    target = np.diag(np.diag(sample_cov))

    # Ledoit-Wolf shrinkage intensity.
    shrinkage = _ledoit_wolf_shrinkage(residuals, sample_cov, target)
    w_matrix = (1.0 - shrinkage) * sample_cov + shrinkage * target

    # Regularise to ensure invertibility.
    w_matrix += np.eye(n_all) * 1e-8

    try:
        w_inv = np.linalg.inv(w_matrix)
    except np.linalg.LinAlgError as exc:
        raise ReconciliationError(
            "covariance matrix is singular; MinT reconciliation not possible"
        ) from exc

    stw = s_matrix.T @ w_inv
    stws = stw @ s_matrix
    try:
        stws_inv = np.linalg.inv(stws)
    except np.linalg.LinAlgError as exc:
        raise ReconciliationError(
            "S' W^{-1} S is singular; MinT reconciliation not possible"
        ) from exc

    p_matrix = stws_inv @ stw
    return s_matrix @ p_matrix @ y_hat


def _ledoit_wolf_shrinkage(
    residuals: np.ndarray,
    sample_cov: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute Ledoit-Wolf shrinkage intensity.

    Simplified oracle-approximation formula.
    """
    n_all, horizon = residuals.shape
    if horizon < 2:
        return 1.0

    # Centre residuals.
    centred = residuals - residuals.mean(axis=1, keepdims=True)

    # Sum of squared off-diagonal elements of sample_cov.
    delta = sample_cov - target
    num = float(np.sum(delta**2))

    if num < 1e-12:
        return 0.0

    # Asymptotic estimate of the optimal shrinkage.
    denom = num * horizon
    pi_hat = 0.0
    for t in range(horizon):
        x_t = centred[:, t : t + 1]
        m_t = x_t @ x_t.T - sample_cov
        pi_hat += float(np.sum(m_t**2))

    shrinkage = pi_hat / denom
    return float(np.clip(shrinkage, 0.0, 1.0))


# ------------------------------------------------------------------
# Coherency checking
# ------------------------------------------------------------------


def check_coherency(
    response: ForecastResponse,
    spec: HierarchySpec,
    tolerance: float = 1e-6,
) -> list[str]:
    """Check whether forecasts satisfy the hierarchical constraints.

    Args:
        response: Forecast response to check.
        spec: Hierarchy specification.
        tolerance: Absolute tolerance for constraint violations.

    Returns:
        A list of violation descriptions.  Empty means the forecasts are
        coherent with respect to the hierarchy.
    """
    fc_map: dict[str, SeriesForecast] = {fc.id: fc for fc in response.forecasts}
    violations: list[str] = []

    for parent_id, children in spec.tree.items():
        if parent_id not in fc_map:
            violations.append(f"parent series {parent_id!r} not found in response")
            continue

        parent_vals = np.array([float(v) for v in fc_map[parent_id].mean])
        child_sum = np.zeros_like(parent_vals)

        missing_children: list[str] = []
        for cid in children:
            if cid not in fc_map:
                missing_children.append(cid)
                continue
            child_vals = np.array([float(v) for v in fc_map[cid].mean])
            if len(child_vals) != len(parent_vals):
                violations.append(
                    f"horizon mismatch: {parent_id!r} has {len(parent_vals)} steps, "
                    f"child {cid!r} has {len(child_vals)} steps"
                )
                continue
            child_sum += child_vals

        if missing_children:
            violations.append(
                f"children of {parent_id!r} missing from response: {missing_children}"
            )
            continue

        diff = np.abs(parent_vals - child_sum)
        max_diff = float(np.max(diff))
        if max_diff > tolerance:
            violating_steps = [int(i) for i in range(len(diff)) if diff[i] > tolerance]
            violations.append(
                f"{parent_id!r} != sum of children at steps {violating_steps} "
                f"(max diff={max_diff:.6g})"
            )

    return violations
