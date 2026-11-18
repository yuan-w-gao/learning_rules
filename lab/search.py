import itertools
import time

from .rules import Cond, Eff, Rule, size
from .sim import RESET, from_obs, physics

# Search for the rules behind ONE visible variable (D1.open, S1.on, a key's floor/held/gone).
# Everything else is read off the observations (teacher forcing), so a candidate only has to
# explain its own variable. Grammar:
#   WHEN <trigger> [IF up to 2 literals] THEN <one effect on the target> [AFTER 0..max_after]
# and a hypothesis is up to max_rules of those. Results come out shortest first.


class Target:
    def __init__(self, obj, prop):
        self.obj, self.prop = obj, prop
        self.is_key = prop is None

    def __eq__(self, other):
        return isinstance(other, Target) and (self.obj, self.prop) == (other.obj, other.prop)

    def __hash__(self):
        return hash((self.obj, self.prop))

    def __repr__(self):
        return f"Target({self.obj}.{self.prop})"

    def value(self, obs):
        for oid, pos, props in obs[1]:
            if oid == self.obj:
                if self.is_key:
                    return "held" if pos is None else "floor"
                return dict(props).get(self.prop, False)
        return "gone" if self.is_key else False

    def values(self):
        return ("floor", "held", "gone") if self.is_key else (True, False)

    def effects(self):
        if self.is_key:
            return [Eff("take", self.obj, None, None), Eff("remove", self.obj, None, None)]
        return [Eff("set", self.obj, self.prop, True), Eff("set", self.obj, self.prop, False),
                Eff("toggle", self.obj, self.prop, None)]

    def result(self, e):
        if e.kind == "toggle":
            return None
        return {"take": "held", "remove": "gone"}.get(e.kind, e.value)

    def apply(self, v, e):
        if e.kind == "toggle":
            return not v
        if e.kind == "take":
            return "held" if v == "floor" else v
        return self.result(e)

    def reads_self(self, c):
        if self.is_key:
            return c.kind == "holding" and c.obj == self.obj
        return c.kind == "prop" and c.obj == self.obj and c.prop == self.prop


DOOR = Target("D1", "open")


def targets(init):
    out = []
    for o in sorted(init.objs.values(), key=lambda o: (o.type != "door", o.id)):
        if o.type == "door":
            out.append(Target(o.id, "open"))
        elif o.type in ("switch", "lamp"):
            out.append(Target(o.id, "on"))
        elif o.type == "key":
            out.append(Target(o.id, None))
    return out


class Search:
    def __init__(self, init, target, max_conds=2, max_after=10, max_rules=3, cap=2000, budget=2_000_000,
                 need_opener=False):
        self.init, self.target = init, target
        self.max_conds, self.max_after, self.max_rules = max_conds, max_after, max_rules
        self.cap, self.budget, self.need_opener = cap, budget, need_opener
        self.triggers = [("tick", None)]
        self.literals = []
        for o in sorted(init.objs.values(), key=lambda o: o.id):
            if o.type == "door":
                self.triggers.append(("bump", o.id))
                self.literals += [Cond("prop", o.id, "open", v) for v in (True, False)]
                continue
            if o.type != "box":
                self.triggers += [("interact", o.id), ("enter", o.id)]
            if o.type == "plate":
                self.triggers += [("pressed", o.id), ("released", o.id)]
                self.literals += [Cond("occupied", o.id, None, v) for v in (True, False)]
            if o.type in ("switch", "lamp"):
                self.literals += [Cond("prop", o.id, "on", v) for v in (True, False)]
            if o.type == "key":
                self.literals += [Cond("holding", o.id, None, v) for v in (True, False)]

    def holds(self, s, c, v):
        t = self.target
        if t.reads_self(c):
            return (v == "held") == bool(c.value) if t.is_key else v == c.value
        if c.kind == "holding":
            return (c.obj in s.inv) == bool(c.value)
        o = s.objs.get(c.obj)
        if c.kind == "occupied":
            return (o is not None and o.pos is not None and s.occupied(o.pos)) == bool(c.value)
        return o is not None and o.props.get(c.prop, False) == c.value

    def tables(self, actions, observations, initial_obs):
        n = len(actions)
        t = self.target
        fired = {trig: 0 for trig in self.triggers}
        lit_ev = {c: 0 for c in self.literals}
        lit_tick = {c: 0 for c in self.literals}
        is_val = {v: 0 for v in t.values()}
        changed = 0
        seg, segs, events_at = 0, [], []
        for i, (a, obs) in enumerate(zip(actions, observations)):
            prev = observations[i - 1] if i else initial_obs
            pv, cv = t.value(prev), t.value(obs)
            is_val[cv] |= 1 << i
            if a == RESET:
                seg += 1
                segs.append(seg)
                events_at.append([])
                continue
            segs.append(seg)
            if cv != pv:
                changed |= 1 << i
            s = from_obs(self.init, prev)
            plates = [o for o in s.objs.values() if o.type == "plate"]
            occ = {p.id: s.occupied(p.pos) for p in plates}
            evs = physics(s, a)
            for p in plates:
                now = s.occupied(p.pos)
                if now != occ[p.id]:
                    evs.append(("pressed" if now else "released", p.id))
            evs.append(("tick", None))
            events_at.append(evs)
            for ev in evs:
                if ev in fired:
                    fired[ev] |= 1 << i
            after = from_obs(self.init, obs)
            for c in self.literals:
                if self.holds(s, c, pv):
                    lit_ev[c] |= 1 << i
                if self.holds(after, c, pv):
                    lit_tick[c] |= 1 << i
        resets = [i for i, a in enumerate(actions) if a == RESET]
        valid = {}
        for k in range(self.max_after + 1):
            m = 0
            for i in range(n - k):
                if segs[i + k] == segs[i] and not any(i < r <= i + k for r in resets):
                    m |= 1 << i
            valid[k] = m
        return dict(n=n, fired=fired, lit_ev=lit_ev, lit_tick=lit_tick, is_val=is_val, changed=changed,
                    valid=valid, events_at=events_at)

    def search(self, actions, observations, initial_obs):
        t0 = time.time()
        tb = self.tables(actions, observations, initial_obs)
        n, changed = tb["n"], tb["changed"]
        full = (1 << n) - 1
        t = self.target

        cond_sets = [()]
        for k in range(1, self.max_conds + 1):
            for combo in itertools.combinations(self.literals, k):
                if len({(c.kind, c.obj, c.prop) for c in combo}) == k:
                    cond_sets.append(combo)

        safe = []  # (rule, cover, size)
        for trig in self.triggers:
            lits = tb["lit_tick"] if trig[0] == "tick" else tb["lit_ev"]
            for cs in cond_sets:
                m = tb["fired"][trig]
                for c in cs:
                    m &= lits[c]
                for after in range(self.max_after + 1):
                    land = ((m & tb["valid"][after]) << after) & full
                    for e in t.effects():
                        r = t.result(e)
                        if r is None:
                            ok, cover = not (land & ~changed), land & changed
                        else:
                            ok, cover = not (land & ~tb["is_val"][r]), land & changed & tb["is_val"][r]
                        if ok:
                            rule = Rule(trig[0], trig[1], cs, (e,), after)
                            safe.append((rule, cover, size(rule)))
        safe.sort(key=lambda x: x[2])

        out = []
        if not changed and not self.need_opener:
            out.append(())
        by_size = {}
        for i, (_, _, sz) in enumerate(safe):
            by_size.setdefault(sz, []).append(i)
        opens = [t.result(r.effects[0]) in (None, True) for r, _, _ in safe]
        self.iters = 0
        checked = 0
        truncated = False

        def ok(members):
            if self.need_opener and not any(opens[i] for i in members):
                return False
            cov = 0
            for i in members:
                cov |= safe[i][1]
            if cov & changed != changed:
                return False
            return len(members) == 1 or self.replay([safe[i][0] for i in members], actions, observations,
                                                    initial_obs, tb["events_at"])

        top = max(by_size) * self.max_rules if by_size else 0
        for d in range(2, top + 1):
            for k in range(1, self.max_rules + 1):
                for members in self.combos(k, d, by_size, safe, changed):
                    checked += 1
                    if checked > self.budget:
                        truncated = True
                        break
                    if ok(members):
                        out.append(tuple(safe[i][0] for i in members))
                        if len(out) >= self.cap:
                            truncated = True
                            break
                if truncated or self.iters > self.budget:
                    truncated = True
                    break
            if truncated:
                break
        self.stats = dict(safe=len(safe), checked=checked, truncated=truncated, seconds=round(time.time() - t0, 2))
        return out

    def combos(self, k, d, by_size, safe, changed):
        for parts in itertools.combinations_with_replacement(sorted(by_size), k):
            if sum(parts) != d:
                continue
            for members in itertools.product(*(by_size[p] for p in parts)):
                self.iters += 1
                if self.iters > self.budget:
                    return
                if any(members[j] >= members[j + 1] for j in range(k - 1)):
                    continue
                # every rule but one has to explain some change (the odd one out can be a needed opener)
                if k > 1 and changed and sum(1 for m in members if not safe[m][1]) > 1:
                    continue
                yield members

    def replay(self, rules, actions, observations, initial_obs, events_at):
        t = self.target
        pending = []
        for i, (a, obs) in enumerate(zip(actions, observations)):
            prev = observations[i - 1] if i else initial_obs
            pv, cv = t.value(prev), t.value(obs)
            if a == RESET:
                pending = []
                continue
            v = pv
            for due, e in pending:
                if due == i:
                    v = t.apply(v, e)
            pending = [p for p in pending if p[0] != i]
            s = from_obs(self.init, prev)
            physics(s, a)
            after = from_obs(self.init, obs)
            for ev, oid in events_at[i]:
                st = after if ev == "tick" else s
                for r in rules:
                    if r.event == ev and r.obj == oid and all(self.holds(st, c, pv) for c in r.conds):
                        if r.after:
                            pending.append((i + r.after, r.effects[0]))
                        else:
                            v = t.apply(v, r.effects[0])
            if v != cv:
                return False
        return True


def door_search(init, cap=300, **kw):
    return Search(init, DOOR, need_opener=True, cap=cap, **kw)
