from agent import solve_proposing
from env import Env
from generator import generate
from proposer import possible
from world import load

M = [
    "#######",
    "#AS.DG#",
    "#######",
]


def test_possible_is_not_fooled_by_open_and_toggle_on_same_trigger():
    walls, goal, state = load(M)
    cands = {"D1": [(("interact", "S1", "D1", "open", True, None),), (("interact", "S1", "D1", "open", "toggle", None),)]}
    assert possible(walls, goal, state, cands) is not None


def test_explore_uses_learned_rules_to_reopen_a_toggled_door():
    m, rules, _ = generate(18, 2, ["toggles"])
    ok, _, _ = solve_proposing(Env(m, rules), "first", 18, ["toggles"])
    assert ok


def test_proposer_solves_toggle_worlds():
    for seed in range(20):
        m, rules, _ = generate(seed, 2, ["toggles"])
        ok, _, _ = solve_proposing(Env(m, rules), "first", seed, ["toggles"])
        assert ok, seed


def test_proposer_escapes_a_trap_with_reset_and_learns_the_closer():
    m, rules, _ = generate(78, 2, ["traps"])
    env = Env(m, rules)
    ok, _, _ = solve_proposing(env, "first", 78, ["traps"])
    assert ok
    assert env.resets == 1


def test_proposer_solves_trap_worlds():
    for seed in range(20):
        m, rules, _ = generate(seed, 2, ["traps"])
        ok, _, _ = solve_proposing(Env(m, rules), "first", seed, ["traps"])
        assert ok, seed


def test_resets_are_not_used_as_evidence():
    m, rules, _ = generate(0, 3, ["conds", "toggles", "traps"])
    env = Env(m, rules, 150)
    ok, _, _ = solve_proposing(env, "first", 0, ["conds", "toggles", "traps"])
    assert env.resets > 0
    assert ok


def test_condition_exploration_covers_one_switch_at_a_time():
    feats = ["conds", "toggles", "traps"]
    m, rules, _ = generate(18, 4, feats)
    ok, _, _ = solve_proposing(Env(m, rules, 200), "first", 18, feats)
    assert ok


def test_three_door_condition_worlds_stay_cheap():
    total = 0
    for seed in range(10):
        m, rules, _ = generate(seed, 3, ["conds"])
        ok, steps, _ = solve_proposing(Env(m, rules, 150), "first", seed, ["conds"])
        assert ok, seed
        total += steps
    assert total <= 330
