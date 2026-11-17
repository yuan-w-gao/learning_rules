def conditions(objs):
    out = []
    for k in objs:
        if objs[k]["type"] == "switch":
            out.append((k, "on", True))
            out.append((k, "on", False))
    return out


def candidates(objs, conds=False):
    doors = [k for k in objs if objs[k]["type"] == "door"]
    others = [k for k in objs if objs[k]["type"] != "door"]
    options = []
    for door in doors:
        rules = []
        for event in ["interact", "enter"]:
            for name in others:
                rules.append((event, name, door, "open", True, None))
                if conds:
                    for c in conditions(objs):
                        rules.append((event, name, door, "open", True, c))
        options.append(rules)
    return options
