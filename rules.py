FEATURES = ("conds", "toggles", "traps")


def features(names):
    names = frozenset(names)
    unknown = names - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown features: {sorted(unknown)}")
    return names


def conditions(objs):
    out = []
    for k in objs:
        if objs[k]["type"] == "switch":
            out.append((k, "on", True))
            out.append((k, "on", False))
    return out


def candidates(objs, feats=()):
    feats = features(feats)
    doors = [k for k in objs if objs[k]["type"] == "door"]
    others = [k for k in objs if objs[k]["type"] != "door"]
    options = []
    for door in doors:
        rules = []
        effects = [True, "toggle"] if "toggles" in feats else [True]
        for event in ["interact", "enter"]:
            for name in others:
                for value in effects:
                    rules.append((event, name, door, "open", value, None))
                    if "conds" in feats:
                        for c in conditions(objs):
                            rules.append((event, name, door, "open", value, c))
        options.append(rules)
    return options
