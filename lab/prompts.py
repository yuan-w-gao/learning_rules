from .rules import EVENTS
from .sim import MOVES, RESET, changes

LEGEND = "@ agent, # wall, G goal, + door (closed) / _ (open), s switch, o plate, B box, k key, l lamp; capitals mean on"


def world_text(world):
    w, h = world["width"], world["height"]
    grid = [["." for _ in range(w)] for _ in range(h)]
    for x, y in world["walls"]:
        grid[y][x] = "#"
    gx, gy = world["goal"]
    grid[gy][gx] = "G"
    for o in world["objects"]:
        x, y = o["pos"]
        on = (o.get("props") or {}).get("on")
        ch = {"door": "+", "switch": "s", "plate": "o", "box": "B", "key": "k", "lamp": "l"}[o["type"]]
        grid[y][x] = ch.upper() if on else ch
    ax, ay = world["agent"]
    grid[ay][ax] = "@"
    lines = [f"Grid {w}x{h}, x grows right and y grows down.",
             f"Agent starts at {tuple(world['agent'])}, goal at {tuple(world['goal'])}.",
             f"Map ({LEGEND}):"]
    lines += ["".join(r) for r in grid]
    lines.append("Objects:")
    lines += [f"- {o['id']}: {o['type']} at {tuple(o['pos'])}" for o in world["objects"]]
    return "\n".join(lines)


def history_text(h):
    if not h.actions:
        return "(no actions taken yet)"
    out = []
    run_start, run_moves = None, []
    for i, (a, obs) in enumerate(zip(h.actions, h.observations)):
        before = h.before(i)
        diff = [c for c in changes(before, obs) if not c.startswith("agent moved")]
        landed = [oid for oid, pos, _ in obs[1] if pos == obs[0]]
        plain = a in MOVES and obs[0] != before[0] and not diff and not landed
        if plain:
            if run_start is None:
                run_start = i
            run_moves.append(a)
            continue
        if run_start is not None:
            out.append(moves_line(run_start, i - 1, run_moves, h.before(i)[0]))
            run_start, run_moves = None, []
        n = i + 1
        if a == RESET:
            out.append(f"step {n}: reset the world to its initial state")
            continue
        if a in MOVES and obs[0] == before[0]:
            what = f"tried to move {a} from {before[0]} but was blocked"
        elif a in MOVES:
            onto = ", ".join(landed) or "nothing"
            what = f"moved {a} onto {onto} at {obs[0]}"
        elif a == "interact":
            what = f"interact at {obs[0]}"
        else:
            what = a
        out.append(f"step {n}: {what} -> " + ("; ".join(diff) if diff else "nothing else changed"))
    if run_start is not None:
        out.append(moves_line(run_start, len(h.actions) - 1, run_moves, h.last[0]))
    return "\n".join(out)


def moves_line(a, b, moves, at):
    span = f"step {a + 1}" if a == b else f"steps {a + 1}-{b + 1}"
    return f"{span}: moved {', '.join(moves)}; now at {at}"


PHYSICS = f"""Known physics:
- Actions: up, down, left, right, interact, wait, and reset (restores the start, costs one step).
- Walls and closed doors block movement. Walking into a closed door is a `bump` event on it.
- Walking into a box pushes it if the cell behind is free.
- `interact` acts on objects in the agent's own cell.
- A plate is occupied when the agent or a box stands on it.

Mechanisms are rules: WHEN <event> [IF <conditions>] THEN <effects> [AFTER k].
Events: {", ".join(EVENTS)} (interact X, enter X, bump D, pressed P / released P for plates, tick every step).
Conditions: X.prop=value, [not] occupied P, [not] holding K.
Effects: set X.prop=value, toggle X.prop, take K, remove X.
Every rule on one event reads the state from before any of them fired."""

REFLECT = """You are exploring a grid world whose objects follow hidden rules. Door D1 stands between you and \
the goal. Reach the goal in as few steps as possible: every step counts, experiments included.

""" + PHYSICS + """

You keep notes between decisions. Each turn you reflect on what just happened, rewrite your notes, \
and choose the next option from a menu of macro-actions."""

REFLECT_UNKNOWN = REFLECT.replace("hidden rules", "unknown rules")
# Opus refused the notes/reflection format as "reasoning extraction"; the policy variant asks for the choice only
POLICY = REFLECT_UNKNOWN.split("You keep notes")[0] + \
    "Look at everything that has happened so far and choose the next option from the menu of macro-actions."


SCORER_HEADER = ("A grid world hides how its objects work. Door D1 blocks the goal. Below are the world, what the "
                 "agent has done and seen so far, and one candidate mechanism for door D1 that reproduces every door "
                 "observation in the history. Most such candidates are coincidences. Decide whether this one is the "
                 "true mechanism.")

DOOR_HEADER = """Write the rules that control door D1 in this grid world: the rules that set D1.open. \
One rule per line, at most 3 lines, then a line END.

Syntax: WHEN <event> [IF <cond> AND <cond>] THEN <effect>, <effect> [AFTER k]
Events: interact X, enter X, bump D1, pressed P, released P, tick
Conditions: X.prop=value, [not] occupied X, [not] holding X
Effects: set D1.open=true, set D1.open=false, toggle D1.open
Example: WHEN interact S2 IF L1.on=true THEN set D1.open=true"""


def door_prompt(world_txt, history_txt):
    return f"{DOOR_HEADER}\n\n# World\n{world_txt}\n\n# History\n{history_txt}\n\n# Door mechanism\n"


PROPOSE = """You are a scientist working out the hidden rules of a grid world from experiments.

""" + PHYSICS + """

Write rules in the text form above, one per line. You may invent hidden props (plain identifiers like armed).
Good hypotheses are COMPLETE (they explain everything seen), they DISAGREE with each other about something \
you could test, and they are ACTIONABLE (they say how D1 can be opened). Prefer simple explanations, \
plus one less obvious one. Never repeat a hypothesis."""


def propose_schema(n):
    return {
        "type": "object", "additionalProperties": False, "required": ["analysis", "hypotheses"],
        "properties": {
            "analysis": {"type": "string"},
            "hypotheses": {"type": "array", "minItems": 1, "maxItems": n, "items": {
                "type": "object", "additionalProperties": False, "required": ["rationale", "rules"],
                "properties": {"rationale": {"type": "string"}, "rules": {"type": "string"}}}},
        },
    }
