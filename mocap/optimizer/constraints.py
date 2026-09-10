# mocap/optimizer/constraints.py

class ConstraintsFilter:
    """Enforces hard constraints: estimated cost <= budget and optional latency <= deadline."""
    def __init__(self, budget: float, deadline: float = None):
        self.budget = budget
        self.deadline = deadline

    def filter_feasible_plans(self, candidate_plans, plan_metrics_map):
        feasible = []
        for plan in candidate_plans:
            metrics = plan_metrics_map.get(plan.plan_id)
            if not metrics:
                continue
            
            # Hard budget rule: estimated cost <= B
            cost_valid = metrics.estimated_cost <= self.budget
            
            # Optional latency rule: estimated latency <= D
            latency_valid = True
            if self.deadline and metrics.estimated_latency:
                latency_valid = metrics.estimated_latency <= self.deadline
            
            if cost_valid and latency_valid:
                feasible.append((plan, metrics))
        return feasible

    