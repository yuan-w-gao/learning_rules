import json
import math

from .agents import Explorer, History, result
from .llm import LLMError
from .prompts import PROPOSE, door_prompt, history_text, propose_schema, world_text
from .rules import RuleError, parse, rule_text, total_size
from .sim import RESET, bfs, changes, from_obs, observe, step
from .verify import consistent, correct

END = "END"
REWARD = {"correct": 1.0, "consistent": 0.2, "none": 0.0}


def parse_door(text):
    body = text.split(END)[0]
    try:
        rules = parse(body)
    except RuleError:
        return None
    if not 1 <= len(rules) <= 3:
        return None
    if any(e.obj != "D1" or e.prop != "open" for r in rules for e in r.effects):
        return None
    return rules


def judge(world, actions, text):
    rules = parse_door(text)
    if rules is None:
        return dict(parsed=False, consistent=False, correct=False, reward=REWARD["none"])
    ok = consistent(world, rules, actions)
    right = ok and correct(world, rules)
    reward = REWARD["correct"] if right else REWARD["consistent"] if ok else REWARD["none"]
    return dict(parsed=True, consistent=ok, correct=right, reward=reward)


def door_prompt_for(world, h):
    return door_prompt(world_text(world), history_text(h))


# --- the LLM hypothesis-pool agent (the first design: the model writes whole rule sets) ---

class Pool:
    def __init__(self, init, lam=0.3):
        self.init, self.lam = init, lam
        self.members = []      # [rules, state, id]
        self.seen = set()
        self.n = 0

    def add(self, rules, h):
        canon = json.dumps(sorted(rule_text(r) for r in rules))
        if canon in self.seen:
            return "duplicate of an earlier hypothesis"
        self.seen.add(canon)
        s = self.init
        for i, (a, obs) in enumerate(zip(h.actions, h.observations)):
            s = self.init if a == RESET else step(s, a, rules)
            if observe(s) != obs:
                bad = "; ".join(changes(observe(s), obs)) or "something else"
                return f"contradicts step {i + 1}: {bad}"
        self.n += 1
        self.members.append([rules, s, f"H{self.n}"])
        return None

    def observe(self, a, obs):
        keep = []
        for m in self.members:
            m[1] = self.init if a == RESET else step(m[1], a, m[0])
            if observe(m[1]) == obs:
                keep.append(m)
        dead = len(self.members) - len(keep)
        self.members = keep
        return dead

    def ranked(self):
        if not self.members:
            return []
        logw = [-self.lam * total_size(m[0]) for m in self.members]
        top = max(logw)
        w = [math.exp(x - top) for x in logw]
        z = sum(w)
        return sorted(zip(self.members, (x / z for x in w)), key=lambda mw: -mw[1])


class PoolAgent:
    def __init__(self, llm, budget=100, max_calls=10, n=5, min_pool=2, retries=2, plan_nodes=50_000, seed=0):
        self.llm, self.budget, self.max_calls, self.n = llm, budget, max_calls, n
        self.min_pool, self.retries, self.plan_nodes, self.seed = min_pool, retries, plan_nodes, seed

    def propose(self, world_txt, ids, pool, h, rejected):
        listing = "\n".join(f"{m[2]}:\n" + "\n".join(rule_text(r) for r in m[0]) for m in pool.members) or "(empty)"
        rej = "\n".join(rejected[-6:]) or "(none)"
        user = (f"# World\n{world_txt}\n\n# History\n{history_text(h)}\n\n"
                f"# Hypotheses still in the pool (do not repeat these)\n{listing}\n\n"
                f"# Your previous proposals that were rejected\n{rej}\n\n"
                f"# Task\nPropose {self.n} complete, mutually distinguishable hypotheses. First, in `analysis`, "
                f"reason about what the history shows. Then, for each hypothesis, a one-sentence rationale and its rules.")
        out = self.llm.complete_json(PROPOSE, user, propose_schema(self.n))
        admitted = 0
        for hyp in out.get("hypotheses", []):
            try:
                rules = parse(hyp["rules"], ids)
            except (RuleError, KeyError) as e:
                rejected.append(f"malformed: {e}")
                continue
            why = pool.add(rules, h)
            if why is None:
                admitted += 1
            elif not why.startswith("duplicate"):
                rejected.append(f"{hyp['rules']!r} -> {why}")
        return admitted

    def run(self, env):
        init = env.initial_state
        world = env.public_world
        world_txt, ids = world_text(world), {o["id"] for o in world["objects"]}
        h = History(env.initial_obs)
        pool = Pool(init)
        explorer = Explorer(self.seed, context=True)
        calls, rejected = 0, []

        def refill():
            nonlocal calls
            tries = 0
            while len(pool.members) < self.min_pool and calls < self.max_calls and tries <= self.retries:
                calls += 1
                try:
                    got = self.propose(world_txt, ids, pool, h, rejected)
                except LLMError:
                    got = 0
                if got == 0:
                    tries += 1
                    if pool.members:
                        break

        refill()
        queue = []
        while not env.done and env.steps < self.budget:
            if not queue:
                now = from_obs(init, h.last)
                for m, _ in pool.ranked():
                    plan = bfs(m[1], m[0], max_nodes=self.plan_nodes)
                    if plan is None and env.steps:
                        fresh = bfs(init, m[0], max_nodes=self.plan_nodes)
                        plan = None if fresh is None else [RESET] + fresh
                    if plan:
                        queue = plan
                        break
                if not queue:
                    queue = bfs(now, (), max_nodes=self.plan_nodes) or [explorer.next_action(now)[0]]
            a = queue.pop(0)
            explorer.record(from_obs(init, h.last), a)
            obs, _ = env.step(a)
            h.add(a, obs)
            dead = pool.observe(a, obs)
            if dead:
                queue = []
            if len(pool.members) < self.min_pool and (dead or not pool.members):
                refill()
        best = pool.ranked()[0][0][0] if pool.members else None
        return result(env, llm_calls=calls, input_tokens=self.llm.usage.input_tokens,
                      output_tokens=self.llm.usage.output_tokens, map_rules=list(best) if best else None)
