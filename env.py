from world import clone, load, step


class BudgetExceeded(Exception):
    pass


class Env:
    def __init__(self, m, rules, budget=100):
        self._walls, self._goal, self._start = load(m)
        self._state = clone(self._start)
        self._rules = rules
        self.budget = budget
        self.steps = 0
        self.resets = 0

    def public(self):
        return frozenset(self._walls), self._goal, clone(self._state)

    def step(self, action):
        if self.steps >= self.budget:
            raise BudgetExceeded(f"budget of {self.budget} steps used up")
        if action == "reset":
            self._state = clone(self._start)
            self.resets += 1
        else:
            self._state = step(self._walls, self._goal, self._state, action, self._rules)
        self.steps += 1
        return clone(self._state)
