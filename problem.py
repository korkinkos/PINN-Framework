class Problem:
    def __init__(self, constraints):
        self.constraints = {}

        for constraint in constraints:
            if constraint.name in self.constraints:
                raise ValueError(
                    f"Duplicate constraint: {constraint.name}"
                )
            self.constraints[constraint.name] = constraint

    def add_constraint(self, constraint):
        if constraint.name in self.constraints:
            raise ValueError(
                f"Duplicate constraint: {constraint.name}"
            )
        self.constraints[constraint.name] = constraint