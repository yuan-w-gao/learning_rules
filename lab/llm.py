import fcntl
import hashlib
import json
import os
import random
import time
from pathlib import Path

# USD per 1M tokens: input, output, cache write, cache read
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.125, 0.01),
}
BATCH_DISCOUNT = 0.5


class LLMError(RuntimeError):
    # the model gave nothing usable: refusal, truncation, bad JSON. Agents fall back on this.
    pass


class BudgetExceeded(RuntimeError):
    # deliberately not an LLMError: running out of money should stop the run, not trigger a fallback
    pass


class Usage:
    def __init__(self):
        self.calls = self.input_tokens = self.output_tokens = 0

    def add(self, i, o):
        self.calls += 1
        self.input_tokens += i
        self.output_tokens += o


def cost_usd(model, i, o, cache_write=0, cache_read=0, batch=False):
    pi, po, pw, pr = PRICES[model]
    usd = (i * pi + o * po + cache_write * pw + cache_read * pr) / 1e6
    return usd * (BATCH_DISCOUNT if batch else 1)


class Ledger:
    # append-only jsonl shared by every run; the cap is on the whole file
    def __init__(self, path="runs/api/ledger.jsonl", cap_usd=None, run=""):
        self.path, self.cap, self.run = Path(path), cap_usd, run
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    def total(self, run=None):
        with open(self.path) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            rows = [json.loads(l) for l in f if l.strip()]
        return sum(r["usd"] for r in rows if run is None or r.get("run") == run)

    def check(self):
        if self.cap is not None and self.total() >= self.cap:
            raise BudgetExceeded(f"spent ${self.total():.2f} of ${self.cap:.2f}")

    def record(self, model, usd, **fields):
        row = {"time": time.time(), "run": self.run, "model": model, "usd": round(usd, 6), **fields}
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row) + "\n")


def hard_cap(cap):
    # soft cap: no new paid episode starts past `cap`. Hard ceiling: calls stop past this.
    return cap + max(2.0, 0.25 * cap)


def strip_schema(schema):
    drop = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength"}
    if isinstance(schema, dict):
        return {k: strip_schema(v) for k, v in schema.items() if k not in drop}
    if isinstance(schema, list):
        return [strip_schema(x) for x in schema]
    return schema


def check_ranges(value, schema):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"] or "maximum" in schema and value > schema["maximum"]:
            raise LLMError(f"{value} out of range")
    if isinstance(value, dict):
        for k, v in value.items():
            check_ranges(v, schema.get("properties", {}).get(k, {}))
    if isinstance(value, list):
        for v in value:
            check_ranges(v, schema.get("items", {}))


def anthropic_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    path = Path(os.environ.get("ANTHROPIC_API_KEY_FILE", "~/.anthropic_key")).expanduser()
    if path.stat().st_mode & 0o077:
        raise PermissionError(f"{path} is readable by others; chmod 600 it")
    return path.read_text().strip()


class AnthropicLLM:
    def __init__(self, model="claude-opus-5-5", effort="medium", max_tokens=16000, ledger=None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=anthropic_key(), max_retries=6)
        self.model, self.effort, self.max_tokens, self.ledger = model, effort, max_tokens, ledger
        self.usage = Usage()
        self.usd = 0.0
        self.refusals = 0

    def complete_json(self, system, user, schema):
        if self.ledger:
            self.ledger.check()
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": strip_schema(schema)}},
        )
        u = resp.usage
        cw, cr = getattr(u, "cache_creation_input_tokens", 0) or 0, getattr(u, "cache_read_input_tokens", 0) or 0
        usd = cost_usd(self.model, u.input_tokens, u.output_tokens, cw, cr)
        self.usd += usd
        self.usage.add(u.input_tokens, u.output_tokens)
        if self.ledger:  # bill first, then complain about the stop reason
            self.ledger.record(self.model, usd, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                               cache_write=cw, cache_read=cr, effort=self.effort, stop=resp.stop_reason)
        if resp.stop_reason == "refusal":
            self.refusals += 1
            raise LLMError("refused")
        if resp.stop_reason == "max_tokens":
            raise LLMError("ran out of tokens")
        txt = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            out = json.loads(txt)
        except json.JSONDecodeError as e:
            raise LLMError(f"bad json: {e}") from None
        check_ranges(out, schema)
        return out


PRESETS = {
    "ollama": ("http://localhost:11434/v1", None),
    "vllm": ("http://localhost:8000/v1", None),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
}


class OpenAICompatLLM:
    def __init__(self, provider="vllm", model="Qwen/Qwen3-8B", temperature=0.7, max_tokens=16384, seed=None,
                 timeout=900, extra_body=None):
        import openai
        url, key_env = PRESETS[provider]
        self.client = openai.OpenAI(base_url=url, api_key=os.environ.get(key_env, "") if key_env else "none",
                                    timeout=timeout)
        self.model, self.temperature, self.max_tokens, self.seed = model, temperature, max_tokens, seed
        self.extra_body = extra_body or {}
        self.usage = Usage()

    def complete_json(self, system, user, schema):
        import openai
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, seed=self.seed,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": schema, "strict": True}},
                extra_body=self.extra_body,
            )
        except (openai.APIConnectionError, openai.APIStatusError) as e:
            raise LLMError(str(e)) from None
        c = resp.choices[0]
        if resp.usage:
            self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        if c.finish_reason == "length":
            raise LLMError("ran out of tokens")
        txt = c.message.content or ""
        try:
            return json.loads(txt[txt.index("{"):])   # skip any thinking preamble
        except (ValueError, json.JSONDecodeError) as e:
            raise LLMError(f"bad json: {e}") from None


class ScriptedLLM:
    # for tests: hands back canned responses in order (or calls them with (system, user))
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []
        self.usage = Usage()

    def complete_json(self, system, user, schema):
        self.prompts.append((system, user))
        self.usage.add(len(system + user) // 4, 0)
        if not self.responses:
            raise LLMError("script ran out")
        r = self.responses.pop(0)
        return r(system, user) if callable(r) else r


class DryRunLLM:
    # costs nothing: fills the schema with junk and projects what the real call would cost
    def __init__(self, model="claude-opus-5-5", seed=0, output_tokens=1500, chars_per_token=2.3):
        self.model, self.rng, self.out_tokens, self.cpt = model, random.Random(seed), output_tokens, chars_per_token
        self.usage = Usage()
        self.usd = 0.0
        self.refusals = 0

    def fill(self, s):
        t = s.get("type")
        if t == "object":
            return {k: self.fill(v) for k, v in s.get("properties", {}).items()}
        if t == "array":
            return [self.fill(s.get("items", {})) for _ in range(s.get("minItems", 1))]
        if t == "integer":
            return self.rng.randint(s.get("minimum", 0), s.get("maximum", 10))
        if t == "boolean":
            return self.rng.random() < 0.5
        return "dry run"

    def complete_json(self, system, user, schema):
        i = int(len(system + user) / self.cpt)
        self.usage.add(i, self.out_tokens)
        self.usd += cost_usd(self.model, i, self.out_tokens)
        return self.fill(schema)


class CachedLLM:
    # replays earlier answers (and earlier failures) for free; transport errors aren't cached
    def __init__(self, llm, key, cache_dir="runs/api/cache"):
        self.llm, self.key, self.dir = llm, key, Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.usage = Usage()

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def complete_json(self, system, user, schema):
        k = {**self.key, "system": system, "user": user, "schema": schema}
        path = self.dir / (hashlib.sha256(json.dumps(k, sort_keys=True).encode()).hexdigest() + ".json")
        if path.exists():
            self.hits += 1
            self.usage.calls += 1
            stored = json.loads(path.read_text())
            if "error" in stored:
                raise LLMError(stored["error"])
            return stored["response"]
        before = (self.llm.usage.input_tokens, self.llm.usage.output_tokens)
        try:
            out = self.llm.complete_json(system, user, schema)
            stored = {"key": self.key, "response": out}
        except LLMError as e:
            stored, out = {"key": self.key, "error": str(e)}, None
        self.usage.add(self.llm.usage.input_tokens - before[0], self.llm.usage.output_tokens - before[1])
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(stored))
        tmp.replace(path)
        if out is None:
            raise LLMError(stored["error"])
        return out


def make_llm(name, cap_usd=None, run="", **kw):
    if name == "anthropic":
        ledger = Ledger(cap_usd=None if cap_usd is None else hard_cap(cap_usd), run=run)
        llm = AnthropicLLM(ledger=ledger, **kw)
        return CachedLLM(llm, {"model": llm.model, "effort": llm.effort, "max_tokens": llm.max_tokens})
    if name in PRESETS:
        return OpenAICompatLLM(provider=name, **kw)
    raise ValueError(f"unknown llm {name!r}")
