from .agents import Explorer, History, experiments, result
from .llm import LLMError
from .prompts import POLICY, REFLECT, history_text, world_text
from .sim import RESET, bfs, from_obs


class ReflectionAgent:
    # Each decision the LLM sees the world, the history and (optionally) its own notes, and picks from the
    # same macro menu the search agent uses. write_notes=False is the choice-only variant.
    def __init__(self, llm, budget=100, system=REFLECT, write_notes=True, max_calls=40, auto_walk=True, seed=0,
                 plan_nodes=20_000):
        self.llm, self.budget, self.system, self.write_notes = llm, budget, system, write_notes
        self.max_calls, self.auto_walk, self.seed, self.plan_nodes = max_calls, auto_walk, seed, plan_nodes
        if not write_notes and system is REFLECT:
            self.system = POLICY

    def menu(self, now, steps):
        opts = []
        path = bfs(now, (), max_nodes=self.plan_nodes)
        if path:
            opts.append(("walk to the goal (reachable now)", path))
        opts += [(m.name, m.actions) for m in experiments(now, nodes=self.plan_nodes)]
        if steps > 0:
            opts.append(("reset the world to its initial state", [RESET]))
        return opts

    def prompt(self, world_txt, h, notes, opts):
        lines = "\n".join(f"{i + 1}. {name} ({max(1, len(acts))} steps)" for i, (name, acts) in enumerate(opts))
        parts = [f"# World\n{world_txt}", f"# History\n{history_text(h)}"]
        if self.write_notes:
            parts.append(f"# Your notes from the last decision\n{notes or '(none yet)'}")
        parts.append(f"# Options from your current position\n{lines}")
        if self.write_notes:
            task = ("In `reflection`, say what the latest actions showed and whether it matched your notes. "
                    "In `notes`, rewrite your notes in full (at most 150 words). "
                    "In `choice`, give the number of the option to take.")
        else:
            task = "In `choice`, give the number of the option to take."
        parts.append("# Task\n" + task)
        return "\n\n".join(parts)

    def schema(self, n):
        props = {"choice": {"type": "integer", "minimum": 1, "maximum": n}}
        if self.write_notes:
            props = {"reflection": {"type": "string"}, "notes": {"type": "string"}, **props}
        return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}

    def run(self, env):
        init = env.initial_state
        world_txt = world_text(env.public_world)
        h = History(env.initial_obs)
        explorer = Explorer(self.seed, context=True)
        notes, calls, errors = "", 0, 0
        trace = []
        while not env.done and env.steps < self.budget:
            now = from_obs(init, h.last)
            opts = self.menu(now, env.steps)
            pick, why = None, ""
            if self.auto_walk and opts and opts[0][0].startswith("walk to the goal"):
                pick, why = opts[0][1], "walk to the goal"
            elif calls < self.max_calls:
                calls += 1
                try:
                    out = self.llm.complete_json(self.system, self.prompt(world_txt, h, notes, opts), self.schema(len(opts)))
                    c = int(out["choice"])
                    if not 1 <= c <= len(opts):
                        raise LLMError(f"choice {c} out of range")
                    pick, why = opts[c - 1][1], opts[c - 1][0]
                    notes = out.get("notes", notes)
                    trace.append({"type": "decide", "t": env.steps, "why": why,
                                  "reflection": out.get("reflection"), "notes": out.get("notes")})
                except (LLMError, KeyError, ValueError, TypeError) as e:
                    errors += 1
                    trace.append({"type": "error", "t": env.steps, "error": str(e)})
            if pick is None:
                a, why = explorer.next_action(now)
                pick = [a]
            for a in pick:
                if env.done or env.steps >= self.budget:
                    break
                explorer.record(from_obs(init, h.last), a)
                obs, _ = env.step(a)
                h.add(a, obs)
        u = self.llm.usage
        return result(env, llm_calls=calls, llm_errors=errors, input_tokens=u.input_tokens,
                      output_tokens=u.output_tokens, usd=round(getattr(self.llm, "usd", 0.0), 4), notes=notes,
                      trace=trace)
