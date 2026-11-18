import random

from .rules import parse, parse_rule, split
from .sim import DEFAULTS, Layout, Obj, State, bfs

PREFIX = {"switch": "S", "plate": "P", "box": "B", "key": "K", "lamp": "L"}


class Reject(Exception):
    pass


class Builder:
    # Random calls happen in the same order as hypo's generator, so (tier, seed, family) gives the same world.
    def __init__(self, rng, width, height):
        self.rng = rng
        self.width, self.height = width, height
        self.cx = rng.randint(3, width - 3)
        door_y = rng.randrange(height)
        self.walls = [(self.cx, y) for y in range(height) if y != door_y]
        self.goal = (rng.randint(self.cx + 1, width - 1), rng.randrange(height))
        self.objects = [{"id": "D1", "type": "door", "pos": (self.cx, door_y)}]
        self.rules = []
        self.counts = {}
        self.taken = {(self.cx - 1, door_y)}
        self.agent = self.free()

    def free(self, interior=False):
        cells = []
        for x in range(self.cx):
            for y in range(self.height):
                if (x, y) in self.taken:
                    continue
                if interior and not (0 < x < self.cx - 1 and 0 < y < self.height - 1):
                    continue
                cells.append((x, y))
        p = self.rng.choice(cells)
        self.taken.add(p)
        return p

    def place(self, kind, interior=False, props=None):
        pre = PREFIX[kind]
        self.counts[pre] = self.counts.get(pre, 0) + 1
        oid = f"{pre}{self.counts[pre]}"
        o = {"id": oid, "type": kind, "pos": self.free(interior)}
        if props:
            o["props"] = props
        self.objects.append(o)
        return oid

    def rule(self, line):
        self.rules.append(line)


# --- families ---

def single_switch(b):
    s = b.place("switch")
    b.rule(f"WHEN interact {s} THEN set {s}.on=true, set D1.open=true")


def toggle_switch(b):
    s = b.place("switch")
    b.rule(f"WHEN interact {s} THEN toggle {s}.on, toggle D1.open")


def latch_plate(b):
    p = b.place("plate")
    b.rule(f"WHEN enter {p} THEN set D1.open=true")


def conjunction(b):
    s1, s2 = b.place("switch"), b.place("switch")
    b.rule(f"WHEN interact {s1} THEN toggle {s1}.on")
    b.rule(f"WHEN interact {s2} THEN toggle {s2}.on")
    b.rule(f"WHEN tick IF {s1}.on=true AND {s2}.on=true THEN set D1.open=true")
    b.rule(f"WHEN tick IF {s1}.on=false THEN set D1.open=false")
    b.rule(f"WHEN tick IF {s2}.on=false THEN set D1.open=false")


def momentary_plate(b):
    p = b.place("plate", interior=True)
    b.place("box", interior=True)
    b.rule(f"WHEN pressed {p} THEN set D1.open=true")
    b.rule(f"WHEN released {p} THEN set D1.open=false")


def key_door(b):
    k = b.place("key")
    b.rule(f"WHEN interact {k} THEN take {k}")
    b.rule(f"WHEN bump D1 IF holding {k} THEN set D1.open=true, remove {k}")


def delayed_open(b):
    s = b.place("switch")
    k = b.rng.randint(2, 4)
    b.rule(f"WHEN interact {s} THEN set {s}.on=true")
    b.rule(f"WHEN interact {s} THEN set D1.open=true AFTER {k}")


def timed_door(b):
    s = b.place("switch")
    k = b.rng.randint(4, 8)
    b.rule(f"WHEN interact {s} THEN set {s}.on=true, set D1.open=true")
    b.rule(f"WHEN interact {s} THEN set {s}.on=false, set D1.open=false AFTER {k}")


def sequence(b):
    s1, s2 = b.place("switch"), b.place("switch")
    b.rule(f"WHEN interact {s1} THEN set {s1}.on=true, set D1.armed=true")
    b.rule(f"WHEN interact {s2} IF D1.armed=true THEN set {s2}.on=true, set D1.open=true")
    b.rule(f"WHEN interact {s2} IF D1.armed=false THEN set {s1}.on=false")


def counter(b):
    s = b.place("switch")
    lamps = [b.place("lamp") for _ in range(b.rng.randint(1, 2))]
    b.rule(f"WHEN interact {s} IF {lamps[0]}.on=false THEN set {s}.on=true, set {lamps[0]}.on=true")
    for prev, nxt in zip(lamps, lamps[1:]):
        b.rule(f"WHEN interact {s} IF {prev}.on=true AND {nxt}.on=false THEN set {nxt}.on=true")
    b.rule(f"WHEN interact {s} IF {lamps[-1]}.on=true THEN set D1.open=true")


def ordered3(b):
    order = [b.place("switch") for _ in range(3)]
    b.rng.shuffle(order)
    a, c, d = order
    off = ", ".join(f"set {s}.on=false" for s in order)
    b.rule(f"WHEN interact {a} THEN set {a}.on=true")
    b.rule(f"WHEN interact {c} IF {a}.on=true THEN set {c}.on=true")
    b.rule(f"WHEN interact {c} IF {a}.on=false THEN {off}")
    b.rule(f"WHEN interact {d} IF {c}.on=true THEN set {d}.on=true, set D1.open=true")
    b.rule(f"WHEN interact {d} IF {c}.on=false THEN {off}")


def inhibitor(b):
    s1, s2 = b.place("switch"), b.place("switch", props={"on": True})
    k = b.rng.randint(6, 8)
    b.rule(f"WHEN interact {s2} THEN set {s2}.on=false")
    b.rule(f"WHEN interact {s2} THEN set {s2}.on=true AFTER {k}")
    b.rule(f"WHEN interact {s1} IF {s2}.on=false THEN set {s1}.on=true, set D1.open=true")


# held out (tier 5): never tune anything on these

def lamp_gate(b):
    s, lamp = b.place("switch"), b.place("lamp")
    b.rule(f"WHEN interact {s} THEN toggle {s}.on, toggle {lamp}.on")
    b.rule(f"WHEN bump D1 IF {lamp}.on=true THEN set D1.open=true")


def box_and_switch(b):
    p = b.place("plate", interior=True)
    b.place("box", interior=True)
    s = b.place("switch")
    b.rule(f"WHEN interact {s} THEN toggle {s}.on")
    b.rule(f"WHEN interact {s} IF occupied {p} THEN set D1.open=true")


def two_keys(b):
    k1, k2 = b.place("key"), b.place("key")
    for k in (k1, k2):
        b.rule(f"WHEN interact {k} THEN take {k}")
    b.rule(f"WHEN bump D1 IF holding {k1} AND holding {k2} THEN set D1.open=true")


def second_press(b):
    s = b.place("switch")
    b.rule(f"WHEN interact {s} THEN set {s}.on=true")
    b.rule(f"WHEN interact {s} IF {s}.on=true THEN set D1.open=true")


def leave_plate(b):
    p, s, lamp = b.place("plate"), b.place("switch"), b.place("lamp")
    b.rule(f"WHEN interact {s} THEN toggle {s}.on, toggle {lamp}.on")
    b.rule(f"WHEN released {p} IF {lamp}.on=true THEN set D1.open=true AFTER {b.rng.randint(1, 3)}")


# tier 6: random door mechanisms for training data only

def random_mechanism(b):
    rng = b.rng
    kinds = rng.sample(["switch", "switch", "plate", "key", "lamp"], rng.randint(1, 3))
    switches, plates, keys, lamps = [], [], [], []
    for kind in kinds:
        if kind == "plate" and rng.random() < 0.4:
            plates.append(b.place("plate", interior=True))
            b.place("box", interior=True)
        elif kind == "switch":
            switches.append(b.place("switch", props={"on": True} if rng.random() < 0.2 else None))
        elif kind == "plate":
            plates.append(b.place("plate"))
        elif kind == "key":
            keys.append(b.place("key"))
        else:
            lamps.append(b.place("lamp"))
    for s in switches:
        r = rng.random()
        if r < 0.6:
            b.rule(f"WHEN interact {s} THEN toggle {s}.on")
        elif r < 0.85:
            b.rule(f"WHEN interact {s} THEN set {s}.on=true")
    for k in keys:
        b.rule(f"WHEN interact {k} THEN take {k}")
    for lamp in lamps:
        ev, src = rng.choice([("interact", s) for s in switches] + [("enter", p) for p in plates] + [("interact", lamp)])
        b.rule(f"WHEN {ev} {src} THEN toggle {lamp}.on")

    triggers = [f"interact {o}" for o in switches + keys + lamps] + [f"enter {p}" for p in plates]
    triggers += [f"{e} {p}" for p in plates for e in ("pressed", "released")] + ["bump D1", "tick"]
    when = rng.choice(triggers)

    def condition():
        opts = [lambda o=o: f"{o}.on={'true' if rng.random() < 0.7 else 'false'}" for o in switches + lamps]
        opts += [lambda p=p: ("" if rng.random() < 0.8 else "not ") + f"occupied {p}" for p in plates]
        opts += [lambda k=k: f"holding {k}" for k in keys]
        if not opts:
            raise Reject("nothing to condition on")
        return rng.choice(opts)()

    n = rng.choice([1, 2]) if when == "tick" else rng.choice([0, 1, 1, 2])
    conds = [condition() for _ in range(n)]
    if len({c.split("=")[0].replace("not ", "") for c in conds}) < len(conds):
        raise Reject("repeated condition")
    after = 0 if rng.random() < 0.7 else rng.randint(1, 5)
    opener = "set D1.open=true" if rng.random() < 0.85 else "toggle D1.open"
    iff = (" IF " + " AND ".join(conds)) if conds else ""
    b.rule(f"WHEN {when}{iff} THEN {opener}" + (f" AFTER {after}" if after else ""))
    r = rng.random()
    if r < 0.2 and opener.startswith("set"):
        b.rule(f"WHEN {when}{iff} THEN set D1.open=false AFTER {after + rng.randint(3, 8)}")
    elif r < 0.35 and conds and "=" in conds[0]:
        var, val = conds[0].split("=")
        b.rule(f"WHEN tick IF {var}={'false' if val == 'true' else 'true'} THEN set D1.open=false")

    types = {o["id"]: o["type"] for o in b.objects}
    rules = parse("\n".join(b.rules))
    if lint(rules, types):
        raise Reject("lint")
    if signature(split(rules)[0], types) in held_out_signatures():
        raise Reject("same structure as a held-out family")


def signature(door_rules, types):
    out = []
    for r in door_rules:
        conds = tuple(sorted((c.kind, types.get(c.obj), str(c.prop), c.value, c.obj == r.obj) for c in r.conds))
        effs = tuple(sorted((e.kind, str(e.value)) for e in r.effects))
        out.append((r.event, types.get(r.obj), conds, effs))
    return tuple(sorted(out))


_held_out = None


def held_out_signatures(seeds=30):
    global _held_out
    if _held_out is None:
        _held_out = set()
        for fam in tier_families(5):
            for seed in range(seeds):
                w = generate(5, seed, fam)
                types = {o["id"]: o["type"] for o in w["objects"]}
                _held_out.add(signature(split(world_rules(w))[0], types))
    return _held_out


def lint(rules, types):
    bad = set()
    for i, r in enumerate(rules):
        if r.event in ("pressed", "released") and types.get(r.obj) != "plate":
            bad.add("dead_event")
        if r.event == "bump" and types.get(r.obj) != "door":
            bad.add("dead_event")
        if r.event == "enter" and types.get(r.obj) == "door" and any(e.obj == r.obj and e.value is True for e in r.effects):
            bad.add("enter_door")
        vals = {}
        for e in r.effects:
            if e.kind == "set":
                vals.setdefault((e.obj, e.prop), set()).add(e.value)
        if any(len(v) > 1 for v in vals.values()):
            bad.add("after_mixed" if r.after else "self_conflict")
        if len(set(r.effects)) != len(r.effects):
            bad.add("repeat")
    writes = lambda r: {(e.obj, e.prop) for e in r.effects if e.kind in ("set", "toggle")}  # noqa: E731
    for a in rules:
        for b in rules:
            if a is b or a.after or (a.event, a.obj, a.conds) == (b.event, b.obj, b.conds) and a == b:
                continue
            if (a.event, a.obj) != (b.event, b.obj):
                continue
            reads = {(c.obj, c.prop) for c in b.conds if c.kind == "prop"}
            if reads & writes(a) & writes(b):
                bad.add("same_snapshot")
    set_vals = {(e.obj, e.prop, e.value) for r in rules for e in r.effects if e.kind == "set"}
    for r in rules:
        for c in r.conds:
            if c.kind == "prop" and not isinstance(c.value, bool) and c.value not in (0, 1) \
                    and (c.obj, c.prop, c.value) not in set_vals:
                bad.add("phantom_value")
    return bad


# --- distractors ---

def inert_switch(b):
    b.place("switch")


def self_switch(b):
    s = b.place("switch")
    b.rule(f"WHEN interact {s} THEN toggle {s}.on")


def lamp_switch(b):
    s, lamp = b.place("switch"), b.place("lamp")
    b.rule(f"WHEN interact {s} THEN toggle {s}.on, toggle {lamp}.on")


def lamp_plate(b):
    p, lamp = b.place("plate"), b.place("lamp")
    b.rule(f"WHEN enter {p} THEN toggle {lamp}.on")


FAMILIES = {
    "single_switch": (1, single_switch), "toggle_switch": (1, toggle_switch), "latch_plate": (1, latch_plate),
    "conjunction": (2, conjunction), "momentary_plate": (2, momentary_plate), "key_door": (2, key_door),
    "delayed_open": (3, delayed_open), "timed_door": (3, timed_door), "sequence": (3, sequence),
    "counter": (4, counter), "ordered3": (4, ordered3), "inhibitor": (4, inhibitor),
    "lamp_gate": (5, lamp_gate), "box_and_switch": (5, box_and_switch), "two_keys": (5, two_keys),
    "second_press": (5, second_press), "leave_plate": (5, leave_plate),
    "random": (6, random_mechanism),
}
DISTRACTORS = {"inert_switch": inert_switch, "self_switch": self_switch, "lamp_switch": lamp_switch,
               "lamp_plate": lamp_plate}
N_DISTRACTORS = {1: (0, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2), 5: (2, 2), 6: (0, 2)}


def tier_families(tier):
    return [f for f, (t, _) in FAMILIES.items() if t == tier]


def tier_of(family):
    return FAMILIES[family][0]


def generate(tier, seed, family=None, tries=50):
    rng = random.Random(f"{tier}-{seed}-{family}")
    family = family or rng.choice(tier_families(tier))
    for _ in range(tries if tier != 6 else 20 * tries):
        w = attempt(rng, tier, seed, family)
        if w is not None:
            return w
    raise RuntimeError(f"couldn't make a solvable {family} world for seed {seed}")


def attempt(rng, tier, seed, family):
    width, height = rng.randint(7, 9), rng.randint(5, 6)
    b = Builder(rng, width, height)
    try:
        FAMILIES[family][1](b)
    except Reject:
        return None
    lo, hi = N_DISTRACTORS[tier]
    names = [rng.choice(sorted(DISTRACTORS)) for _ in range(rng.randint(lo, hi))]
    for d in names:
        DISTRACTORS[d](b)
    w = {
        "name": f"L{tier}-{family}-{seed}", "width": width, "height": height, "walls": b.walls, "goal": b.goal,
        "agent": b.agent, "objects": b.objects, "rules": "\n".join(b.rules),
        "tier": tier, "family": family, "seed": seed, "distractors": names,
    }
    start = initial_state(w)
    rules = world_rules(w)
    sol = bfs(start, rules)
    if sol is None or bfs(start, ()) is not None:
        return None
    w["optimal"] = len(sol)
    return w


def world_rules(w):
    return parse(w["rules"], {o["id"] for o in w["objects"]})


def initial_state(w):
    objs = {}
    for o in w["objects"]:
        props = {**DEFAULTS.get(o["type"], {}), **o.get("props", {})}
        objs[o["id"]] = Obj(o["id"], o["type"], tuple(o["pos"]), props)
    return State(Layout(w["width"], w["height"], w["walls"], w["goal"]), tuple(w["agent"]), objs)


def public(w):
    return {k: w[k] for k in ("name", "width", "height", "walls", "goal", "agent", "objects")}


def draw(w, s=None):
    s = s or initial_state(w)
    rows = []
    for y in range(w["height"]):
        row = ""
        for x in range(w["width"]):
            p = (x, y)
            here = [o for o in s.objs.values() if o.pos == p]
            if p == s.agent:
                row += "@"
            elif p in s.layout.walls:
                row += "#"
            elif here:
                o = here[0]
                ch = {"door": "D", "switch": "S", "plate": "P", "box": "B", "key": "K", "lamp": "L"}[o.type]
                if o.props.get("open") or o.props.get("on"):
                    ch = ch.lower()
                row += ch
            elif p == s.layout.goal:
                row += "G"
            else:
                row += "."
        rows.append(row)
    return "\n".join(rows)
