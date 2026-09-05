# mocap/optimizer/pareto.py

class ParetoSelector:
    """Performs Pareto filtering for multi-objective candidate selection[cite: 1]."""
    @staticmethod
    def get_pareto_optimal(feasible_plans):
        pareto_front = []
        for p_curr, m_curr in feasible_plans:
            dominated = False
            for p_other, m_other in feasible_plans:
                if p_curr.plan_id == p_other.plan_id:
                    continue
                
                cost_other = m_other.estimated_cost
                cost_curr = m_curr.estimated_cost
                lat_other = m_other.estimated_latency or 0.0
                lat_curr = m_curr.estimated_latency or 0.0

                # Check if another plan dominates current plan across cost and latency
                if (cost_other <= cost_curr and lat_other <= lat_curr) and \
                   (cost_other < cost_curr or lat_other < lat_curr):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append((p_curr, m_curr))
        return pareto_front


    