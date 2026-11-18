from collections import deque

MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
ACTIONS = ("up", "down", "left", "right", "interact", "wait")
RESET = "reset"

VISIBLE = {"switch": ("on",), "door": ("open",), "lamp": ("on",), "plate": (), "box": (), "key": ()}
DEFAULTS = {"switch": {"on": False}, "door": {"open": False}, "lamp": {"on": False}}


class Layout:
    __slots__ = ("width", "height", "walls", "goal")

    def __init__(self, width, height, walls, goal):
        self.width, self.height, self.walls, self.goal = width, height, frozenset(map(tuple, walls)), tuple(goal)

    def inside(self, p):
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height


class Obj:
    __slots__ = ("id", "type", "pos", "props")

    def __init__(self, id, type, pos, props):
        self.id, self.type, self.pos, self.props = id, type, pos, props

    def copy(self):
        return Obj(self.id, self.type, self.pos, dict(self.props))


class State:
    __slots__ = ("layout", "agent", "objs", "inv", "pending", "t", "done")

    def __init__(self, layout, agent, objs, inv=frozenset(), pending=(), t=0, done=False):
        self.layout, self.agent, self.objs = layout, agent, objs
        self.inv, self.pending, self.t, self.done = inv, pending, t, done

    def copy(self):
        return State(self.layout, self.agent, {k: o.copy() for k, o in self.objs.items()},
                     self.inv, self.pending, self.t, self.done)

    def at(self, p):
        return [o for o in self.objs.values() if o.pos == p]

    def closed_door(self, p):
        for o in self.objs.values():
            if o.pos == p and o.type == "door" and not o.props.get("open", False):
                return o
        return None

    def box(self, p):
        for o in self.objs.values():
            if o.pos == p and o.type == "box":
                return o
        return None

    def occupied(self, p):
        return self.agent == p or self.box(p) is not None

    def passable(self, p):
        return self.layout.inside(p) and p not in self.layout.walls and self.closed_door(p) is None

    def key(self):
        objs = tuple((o.id, o.pos, tuple(sorted(o.props.items()))) for o in sorted(self.objs.values(), key=lambda o: o.id))
        pending = tuple((due - self.t, effs) for due, effs in self.pending)
        return self.agent, objs, tuple(sorted(self.inv)), pending, self.done


def physics(s, action):
    # mutates s, returns the events
    events = []
    if action == "interact":
        events += [("interact", o.id) for o in s.at(s.agent)]
    elif action in MOVES:
        dx, dy = MOVES[action]
        nxt = (s.agent[0] + dx, s.agent[1] + dy)
        if not s.passable(nxt):
            door = s.closed_door(nxt)
            if door is not None:
                events.append(("bump", door.id))
            return events
        b = s.box(nxt)
        if b is not None:
            beyond = (nxt[0] + dx, nxt[1] + dy)
            if not s.passable(beyond) or s.box(beyond) is not None:
                return events
            b.pos = beyond
        s.agent = nxt
        events += [("enter", o.id) for o in s.at(nxt) if o.type != "box"]
    return events


def holds(s, c):
    if c.kind == "holding":
        return (c.obj in s.inv) == bool(c.value)
    o = s.objs.get(c.obj)
    if c.kind == "occupied":
        return (o is not None and o.pos is not None and s.occupied(o.pos)) == bool(c.value)
    return o is not None and o.props.get(c.prop, False) == c.value


def apply(s, effects):
    for e in effects:
        o = s.objs.get(e.obj)
        if o is None:
            continue
        if e.kind == "set":
            o.props[e.prop] = e.value
        elif e.kind == "toggle":
            o.props[e.prop] = not o.props.get(e.prop, False)
        elif e.kind == "take":
            if o.pos is not None:
                o.pos = None
                s.inv = s.inv | {o.id}
        elif e.kind == "remove":
            del s.objs[o.id]
            s.inv = s.inv - {o.id}


def step(state, action, rules):
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    s = state.copy()
    if s.done:
        return s
    plates = [o for o in s.objs.values() if o.type == "plate"]
    before = {p.id: s.occupied(p.pos) for p in plates}
    events = physics(s, action)
    for p in plates:
        now = s.occupied(p.pos)
        if now and not before[p.id]:
            events.append(("pressed", p.id))
        elif before[p.id] and not now:
            events.append(("released", p.id))
    events.append(("tick", None))

    s.t += 1
    due = [effs for t, effs in s.pending if t <= s.t]
    s.pending = tuple(x for x in s.pending if x[0] > s.t)
    for effs in due:
        apply(s, effs)

    for ev, oid in events:
        fired = [r for r in rules if r.event == ev and r.obj == oid and all(holds(s, c) for c in r.conds)]
        for r in fired:
            if r.after:
                s.pending = s.pending + ((s.t + r.after, r.effects),)
            else:
                apply(s, r.effects)

    if s.agent == s.layout.goal:
        s.done = True
    return s


def observe(s):
    objs = tuple((o.id, o.pos, tuple((p, o.props.get(p, False)) for p in VISIBLE[o.type]))
                 for o in sorted(s.objs.values(), key=lambda o: o.id))
    return s.agent, objs, tuple(sorted(s.inv)), s.done


def from_obs(init, obs):
    # physics-only state that looks like obs: no latent props, nothing pending
    agent, objs, inv, done = obs
    seen = {oid: (pos, props) for oid, pos, props in objs}
    out = {}
    for oid, o in init.objs.items():
        if oid in seen:
            pos, props = seen[oid]
            out[oid] = Obj(oid, o.type, pos, dict(props))
    return State(init.layout, agent, out, frozenset(inv), (), 0, done)


def visible(obs, oid, prop):
    for i, pos, props in obs[1]:
        if i == oid:
            return dict(props).get(prop, False)
    return False


def changes(before, after):
    out = []
    if before[0] != after[0]:
        out.append(f"agent moved {before[0]} -> {after[0]}")
    b = {i: (pos, dict(p)) for i, pos, p in before[1]}
    a = {i: (pos, dict(p)) for i, pos, p in after[1]}
    for oid in sorted(set(a) | set(b)):
        if oid not in a:
            out.append(f"{oid} disappeared")
        elif oid not in b:
            out.append(f"{oid} appeared")
        else:
            (pb, db), (pa, da) = b[oid], a[oid]
            if pb != pa:
                out.append(f"{oid} moved {pb} -> {pa}")
            for k in sorted(set(db) | set(da)):
                if db.get(k) != da.get(k):
                    out.append(f"{oid}.{k}: {fmt_v(db.get(k))} -> {fmt_v(da.get(k))}")
    for oid in sorted(set(after[2]) - set(before[2])):
        out.append(f"picked up {oid}")
    if after[3] and not before[3]:
        out.append("reached the goal")
    return out


def fmt_v(v):
    return str(v).lower() if isinstance(v, bool) else str(v)


def bfs(start, rules, goal=None, max_nodes=200_000):
    goal = goal or (lambda s: s.done)
    if goal(start):
        return []
    parent = {start.key(): None}
    q = deque([start])
    while q and len(parent) < max_nodes:
        s = q.popleft()
        sk = s.key()
        for a in ACTIONS:
            n = step(s, a, rules)
            k = n.key()
            if k in parent:
                continue
            parent[k] = (sk, a)
            if goal(n):
                path = []
                while parent[k] is not None:
                    k, a = parent[k]
                    path.append(a)
                return path[::-1]
            q.append(n)
    return None


def run(s, actions, rules, init=None):
    for a in actions:
        s = init if a == RESET else step(s, a, rules)
    return s
