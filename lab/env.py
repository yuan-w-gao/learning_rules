from .sim import RESET, observe, step
from .worlds import initial_state, public, world_rules


class BudgetExceeded(RuntimeError):
    pass


class Env:
    # Agents get public_world, initial_state, initial_obs and whatever step returns. Nothing else.
    def __init__(self, world, budget=100):
        self._world = world
        self._rules = world_rules(world)
        self.budget = budget
        self._state = initial_state(world)
        self.initial_obs = observe(self._state)
        self.actions = []
        self.observations = []

    @property
    def public_world(self):
        return public(self._world)

    @property
    def initial_state(self):
        return initial_state(self._world)

    @property
    def steps(self):
        return len(self.actions)

    @property
    def done(self):
        return self._state.done

    def step(self, action):
        if self.budget is not None and self.steps >= self.budget:
            raise BudgetExceeded(f"budget of {self.budget} steps used up")
        if action == RESET:
            self._state = initial_state(self._world)
        else:
            self._state = step(self._state, action, self._rules)
        obs = observe(self._state)
        self.actions.append(action)
        self.observations.append(obs)
        return obs, self._state.done
