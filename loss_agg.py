class LossAggregator:
    def compute(self, problem, model, points_by_constraint):
        losses = {}
        total = None

        for name, constraint in problem.constraints.items():
            points = points_by_constraint[name]
            raw_loss = constraint.loss(model, points)
            weighted_loss = constraint.weight * raw_loss

            losses[name] = raw_loss
            total = (
                weighted_loss
                if total is None
                else total + weighted_loss
            )

        if total is None:
            raise ValueError("Problem has no constraints")

        losses["total"] = total
        return losses