import pytest

from agent import solve, solve_proposing
from env import BudgetExceeded, Env
from generator import generate
from rules import candidates
from world import load


def test_generate_seed_0_is_stable():
    m, rules, _ = generate(0)
    assert m == [
        "###############",
        "#..S.A#....#..#",
        "#.....#....D..#",
        "#.P...D....#G.#",
        "#....S#....#..#",
        "###############",
    ]
    assert rules == [("enter", "P1", "D1", "open", True, None), ("interact", "S2", "D2", "open", True, None)]


@pytest.mark.parametrize("doors", [2, 3])
@pytest.mark.parametrize("feats", [(), ("conds",)])
def test_generated_worlds_need_rules(doors, feats):
    from planner import bfs
    for seed in range(10):
        m, rules, _ = generate(seed, doors, feats)
        walls, goal, state = load(m)
        assert bfs(walls, goal, state, rules) is not None
        assert bfs(walls, goal, state, []) is None


def test_candidate_counts():
    objs = load(generate(0)[0])[2]["objs"]
    assert [len(c) for c in candidates(objs)] == [6, 6]
    assert [len(c) for c in candidates(objs, ["conds"])] == [30, 30]


def test_env_budget_and_copies():
    m, rules, _ = generate(0)
    env = Env(m, rules, budget=2)
    walls, goal, state = env.public()
    state["objs"]["D1"]["open"] = True
    assert env.public()[2]["objs"]["D1"]["open"] is False
    env.step("left")
    env.step("left")
    with pytest.raises(BudgetExceeded):
        env.step("left")


@pytest.mark.parametrize(
    "solver, feats, max_total",
    [(solve, (), 252), (solve_proposing, (), 212), (solve_proposing, ("conds",), 322)],
)
def test_agents_solve_first_ten_seeds(solver, feats, max_total):
    total = 0
    for seed in range(10):
        ok, steps, _ = solver(Env(*generate(seed, 2, feats)[:2]), "first", seed, feats)
        assert ok, seed
        total += steps
    assert total <= max_total


def test_reset_restores_world_and_costs_a_step():
    m, rules, _ = generate(0)
    env = Env(m, rules)
    start = env.public()[2]
    env.step("left")
    after = env.step("reset")
    assert after == start
    assert env.steps == 2
    assert env.resets == 1


def test_trap_worlds_have_a_closing_rule_beyond_its_door():
    for seed in range(10):
        m, rules, _ = generate(seed, 2, ["traps"])
        objs = load(m)[2]["objs"]
        event, obj, door, prop, value, cond = rules[-1]
        assert value is False
        assert objs[obj]["pos"][0] > objs[door]["pos"][0]
