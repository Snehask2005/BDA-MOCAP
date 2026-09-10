# mocap/optimizer/lagrangian.py

class LagrangianRanker:
    """Implements Lagrangian ranking and multi-objective optimization[cite: 1]:
       J = alpha*L + beta*M + gamma*A + delta*V
    """
    def __init__(self, alpha=0.25, beta=0.25, gamma=0.25, delta=0.25, lambda_mult=0.0):
        self.alpha = alpha  # Latency weight
        self.beta = beta    # Cost/Resource weight
        self.gamma = gamma  # Accuracy weight
        self.delta = delta  # Variance weight
        self.lambda_mult = lambda_mult  # Lagrangian multiplier

    def compute_score(self, plan, metrics, budget):
        L = metrics.estimated_latency or 0.0
        M = metrics.estimated_cost or 0.0
        A = getattr(metrics, 'accuracy_tolerance', 1.0)
        V = getattr(metrics, 'variance', 0.0)
        
        # Base multi-objective function
        base_objective = (self.alpha * L) + (self.beta * M) + (self.gamma * A) + (self.delta * V)
        
        # Lagrangian relaxation penalty for budget constraint
        lagrangian_penalty = self.lambda_mult * max(0.0, M - budget)
        
        return base_objective + lagrangian_penalty

    def rank_candidates(self, feasible_plans, budget):
        scored = []
        for plan, metrics in feasible_plans:
            score = self.compute_score(plan, metrics, budget)
            scored.append((plan, metrics, score))
        
        # Sort ascending (lower score is better)
        scored.sort(key=lambda x: x[2])
        return scored
