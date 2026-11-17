import pytest

from planner import bfs
from world import clone, key, load, step

M = [
    "#######",
    "#AS.DG#",
    "#..P#.#",
    "#######",
]
OPEN_ON_SWITCH = ("interact", "S1", "D1", "open", True, None)
OPEN_ON_PLATE_IF_ON = ("enter", "P1", "D1", "open", True, ("S1", "on", True))


@pytest.fixture
def world():
    return load(M)


def walk(walls, goal, state, actions, rules=()):
    for a in actions:
        state = step(walls, goal, state, a, rules)
    return state


def test_load(world):
    walls, goal, state = world
    assert state["agent"] == (1, 1)
    assert goal == (5, 1)
    assert sorted(state["objs"]) == ["D1", "P1", "S1"]
    assert state["objs"]["D1"]["open"] is False


def test_walls_and_closed_doors_block(world):
    walls, goal, state = world
    assert walk(walls, goal, state, ["up"])["agent"] == (1, 1)
    assert walk(walls, goal, state, ["right"] * 5)["agent"] == (3, 1)


def test_interact_toggles_switch_and_fires_rule(world):
    walls, goal, state = world
    s = walk(walls, goal, state, ["right", "interact"], [OPEN_ON_SWITCH])
    assert s["objs"]["S1"]["on"] is True
    assert s["objs"]["D1"]["open"] is True


def test_condition_gates_rule(world):
    walls, goal, state = world
    to_plate = ["down", "right", "right"]
    off = walk(walls, goal, state, to_plate, [OPEN_ON_PLATE_IF_ON])
    on = walk(walls, goal, state, ["right", "interact", "left"] + to_plate, [OPEN_ON_PLATE_IF_ON])
    assert off["objs"]["D1"]["open"] is False
    assert on["objs"]["D1"]["open"] is True


def test_reaching_goal_sets_done(world):
    walls, goal, state = world
    s = walk(walls, goal, state, ["right", "interact"] + ["right"] * 3, [OPEN_ON_SWITCH])
    assert s["done"] is True


def test_step_does_not_change_its_input(world):
    walls, goal, state = world
    before = key(state)
    step(walls, goal, state, "right", [])
    assert key(state) == before


def test_clone_is_independent(world):
    _, _, state = world
    c = clone(state)
    c["objs"]["D1"]["open"] = True
    assert state["objs"]["D1"]["open"] is False
    assert key(c) != key(state)


def test_bfs(world):
    walls, goal, state = world
    assert bfs(walls, goal, state, [OPEN_ON_SWITCH]) == ["right", "interact", "right", "right", "right"]
    assert bfs(walls, goal, state, []) is None


def test_toggle_flips_door_each_time(world):
    walls, goal, state = world
    rule = ("interact", "S1", "D1", "open", "toggle", None)
    once = walk(walls, goal, state, ["right", "interact"], [rule])
    twice = walk(walls, goal, once, ["interact"], [rule])
    assert once["objs"]["D1"]["open"] is True
    assert twice["objs"]["D1"]["open"] is False
