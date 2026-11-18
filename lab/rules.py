import re
from collections import namedtuple

EVENTS = ("interact", "enter", "bump", "pressed", "released", "tick")

Cond = namedtuple("Cond", "kind obj prop value")    # kind: prop | occupied | holding
Eff = namedtuple("Eff", "kind obj prop value")      # kind: set | toggle | take | remove
Rule = namedtuple("Rule", "event obj conds effects after")


class RuleError(ValueError):
    pass


def size(rule):
    return 1 + len(rule.conds) + len(rule.effects) + (1 if rule.after else 0)


def total_size(rules):
    return sum(size(r) for r in rules)


def fmt(v):
    return str(v).lower() if isinstance(v, bool) else str(v)


def rule_text(r):
    out = "WHEN " + r.event + (" " + r.obj if r.obj else "")
    if r.conds:
        parts = []
        for c in r.conds:
            if c.kind == "prop":
                parts.append(f"{c.obj}.{c.prop}={fmt(c.value)}")
            else:
                parts.append(("" if c.value else "not ") + f"{c.kind} {c.obj}")
        out += " IF " + " AND ".join(parts)
    effs = []
    for e in r.effects:
        if e.kind == "set":
            effs.append(f"set {e.obj}.{e.prop}={fmt(e.value)}")
        elif e.kind == "toggle":
            effs.append(f"toggle {e.obj}.{e.prop}")
        else:
            effs.append(f"{e.kind} {e.obj}")
    out += " THEN " + ", ".join(effs)
    if r.after:
        out += f" AFTER {r.after}"
    return out


def text(rules):
    return "\n".join(rule_text(r) for r in rules)


def parse_value(tok):
    if tok in ("true", "false"):
        return tok == "true"
    try:
        return int(tok)
    except ValueError:
        raise RuleError(f"bad value {tok!r}") from None


RULE_RE = re.compile(r"WHEN (\w+)(?: (\w+))?(?: IF (.+?))? THEN (.+?)(?: AFTER (\d+))?")


def parse_rule(line, ids=None):
    m = RULE_RE.fullmatch(line.strip())
    if not m:
        raise RuleError(f"can't parse {line!r}")
    event, obj, conds_s, effs_s, after = m.groups()
    if event not in EVENTS:
        raise RuleError(f"unknown event {event!r}")
    if (event == "tick") != (obj is None):
        raise RuleError("tick takes no object, every other event needs one")

    def check(o):
        if ids is not None and o not in ids:
            raise RuleError(f"unknown object {o!r}")
        return o

    if obj:
        check(obj)
    conds = []
    for c in conds_s.split(" AND ") if conds_s else []:
        c = c.strip()
        if m2 := re.fullmatch(r"(\w+)\.(\w+)=(\w+)", c):
            conds.append(Cond("prop", check(m2[1]), m2[2], parse_value(m2[3])))
        elif m2 := re.fullmatch(r"(not )?(occupied|holding) (\w+)", c):
            conds.append(Cond(m2[2], check(m2[3]), None, not m2[1]))
        else:
            raise RuleError(f"can't parse condition {c!r}")
    effs = []
    for e in effs_s.split(","):
        e = e.strip()
        if m2 := re.fullmatch(r"set (\w+)\.(\w+)=(\w+)", e):
            effs.append(Eff("set", check(m2[1]), m2[2], parse_value(m2[3])))
        elif m2 := re.fullmatch(r"toggle (\w+)\.(\w+)", e):
            effs.append(Eff("toggle", check(m2[1]), m2[2], None))
        elif m2 := re.fullmatch(r"(take|remove) (\w+)", e):
            effs.append(Eff(m2[1], check(m2[2]), None, None))
        else:
            raise RuleError(f"can't parse effect {e!r}")
    after = int(after or 0)
    if after > 20:
        raise RuleError("AFTER is at most 20")
    return Rule(event, obj, tuple(conds), tuple(effs), after)


def parse(block, ids=None):
    return tuple(parse_rule(l, ids) for l in block.strip().splitlines() if l.strip())


def english(r):
    trig = {
        "interact": f"the agent interacts with {r.obj}",
        "enter": f"the agent steps onto {r.obj}",
        "bump": f"the agent bumps into {r.obj}",
        "pressed": f"{r.obj} becomes pressed",
        "released": f"{r.obj} is released",
    }.get(r.event)
    s = f"When {trig}" if trig else "Every step"
    if r.conds:
        bits = []
        for c in r.conds:
            neg = "" if c.value else "not "
            if c.kind == "prop":
                bits.append(f"{c.obj}.{c.prop} is {fmt(c.value)}")
            elif c.kind == "occupied":
                bits.append(f"{c.obj} is {neg}occupied")
            else:
                bits.append(f"the agent is {neg}holding {c.obj}")
        s += ", if " + " and ".join(bits)
    effs = []
    for e in r.effects:
        if e.kind == "set":
            effs.append(f"set {e.obj}.{e.prop} to {fmt(e.value)}")
        elif e.kind == "toggle":
            effs.append(f"toggle {e.obj}.{e.prop}")
        elif e.kind == "take":
            effs.append(f"the agent picks up {e.obj}")
        else:
            effs.append(f"{e.obj} disappears")
    s += ", " + ", ".join(effs)
    if r.after:
        s += f" after {r.after} steps"
    return s + "."


def split(rules, obj="D1", prop="open"):
    # everything that writes obj.prop vs the rest, effect by effect (safe: all rules on one event read the same snapshot)
    mine, rest = [], []
    for r in rules:
        a = tuple(e for e in r.effects if e.obj == obj and e.prop == prop)
        b = tuple(e for e in r.effects if not (e.obj == obj and e.prop == prop))
        if a:
            mine.append(r._replace(effects=a))
        if b:
            rest.append(r._replace(effects=b))
    return tuple(mine), tuple(rest)
