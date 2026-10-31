## MOCAP PR Quality Gate Checklist

### 1. Ownership & Work Package
- [ ] Module Owner sign-off: Primary owner + cross-functional peer.
- [ ] Work package: WP2 (Features) / WP3 (Predictor) / WP4 (Conformal) / Other.

### 2. Guardrails Verification
- [ ] **Data Leakage Prevention**: Cross-validation splits keep all runs/plans of a query strictly together in either train or test fold.
- [ ] **Hint Execution Verification**: Verified physical execution plan matches intended hints (`spark.sql.cbo.joinReorder.enabled=false`).
- [ ] **Zero CI Failures**: Automated pytest passes without errors.

### 3. Acceptance Criteria
- [ ] Distinct predicted values > 50 for candidate plan sets (resolving 5-value collapse).
- [ ] Spearman $\rho \ge 0.70$ and Kendall $\tau \ge 0.60$ on held-out test queries.
- [ ] Conformal coverage within $\pm 2\%$ of target $(1 - \delta)$.
