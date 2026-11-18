import math
from concurrent.futures import ThreadPoolExecutor

from .llm import LLMError
from .prompts import PHYSICS, SCORER_HEADER, history_text, world_text
from .rules import english


def scorer_prompt(world_txt, history_txt, door):
    prefix = f"{SCORER_HEADER}\n\n# World\n{world_txt}\n\n# History\n{history_txt}\n\n# Candidate\n"
    return prefix, " AND ".join(english(r) for r in door) + "\n\nIs this the true mechanism? Answer:"


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class LearnedScorer:
    # A LoRA-tuned model served by vLLM that answers " yes"/" no". Score = log P(yes) - log P(no),
    # added to the candidate's prior log-weight by the search agent.
    scores = True

    def __init__(self, world, base_url="http://localhost:8000/v1", model="scorer", workers=32, timeout=120):
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key="none", timeout=timeout)
        self.model, self.workers = model, workers
        self.world_txt = world_text(world)
        self.errors = []

    def one(self, prompt):
        r = self.client.completions.create(model=self.model, prompt=prompt, max_tokens=1, temperature=0, logprobs=20)
        top = r.choices[0].logprobs.top_logprobs[0]
        yes = max((lp for tok, lp in top.items() if tok.strip().lower() == "yes"), default=-30.0)
        no = max((lp for tok, lp in top.items() if tok.strip().lower() == "no"), default=-30.0)
        return yes - no

    def __call__(self, history, doors):
        htxt = history_text(history)
        prompts = ["".join(scorer_prompt(self.world_txt, htxt, d)) for d in doors]
        try:
            with ThreadPoolExecutor(self.workers) as ex:
                return list(ex.map(self.one, prompts))
        except Exception as e:  # a dead server shouldn't kill the episode; the prior still works
            self.errors.append(repr(e))
            return [0.0] * len(doors)


class HeuristicScorer:
    # free stand-in for the learned scorer: favours candidates whose trigger the agent actually touched
    # and that mention objects that visibly changed. Not tuned on held-out families.
    scores = True

    def __call__(self, history, doors):
        touched = set()
        for i, a in enumerate(history.actions):
            if a == "interact":
                touched |= {oid for oid, pos, _ in history.observations[i][1] if pos == history.observations[i][0]}
        out = []
        for d in doors:
            s = 0.0
            for r in d:
                if r.obj in touched:
                    s += 0.5
                if r.event == "tick" and not r.conds:
                    s -= 1.0
            out.append(s)
        return out


RANKER_SYSTEM = """You help a scientific-discovery agent in a grid world. Every candidate you are shown \
reproduces every door observation so far; most are coincidences. Judge which are most likely the real mechanism.

""" + PHYSICS


class LLMRanker:
    scores = False

    def __init__(self, llm, world, k=5):
        self.llm, self.k = llm, k
        self.world_txt = world_text(world)

    def __call__(self, history, doors):
        k = min(self.k, len(doors))
        listing = "\n".join(f"{i + 1}. " + " AND ".join(english(r) for r in d) for i, d in enumerate(doors))
        user = (f"# World\n{self.world_txt}\n\n# History\n{history_text(history)}\n\n"
                f"# Candidate door mechanisms ({len(doors)}), all consistent with the history\n{listing}\n\n"
                f"# Task\nIn `analysis`, reason about which candidates fit what a designer would build. Then in `picks` "
                f"give the {k} most likely candidate numbers, most likely first, preferring candidates that differ.")
        schema = {"type": "object", "additionalProperties": False, "required": ["analysis", "picks"],
                  "properties": {"analysis": {"type": "string"},
                                 "picks": {"type": "array", "minItems": k, "maxItems": k,
                                           "items": {"type": "integer", "minimum": 1, "maximum": len(doors)}}}}
        try:
            out = self.llm.complete_json(RANKER_SYSTEM, user, schema)
        except LLMError:
            return []
        return [p - 1 for p in out.get("picks", [])]
