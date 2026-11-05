"""Data-grounded, tool-forcing task instances.

Why
---
34 of the suite's 73 bundles exercise exactly one tool for the whole session;
langchain is 25 of 26. The cause is task design, not model choice: the runner
issues one generic, knowledge-answerable question per session
("How do agent workloads stress CPU microarchitecture?"), so there is exactly
one sensible move per round, forever. Re-recording with a frontier model makes
it worse -- it answers from knowledge and terminates early, collapsing autogen
from 15 rounds to 1.

A grounded task instance has four properties:

  1. per-round      generated from (seed, round_index), not repeated
  2. unknowable     refers to entities minted at runtime, so parametric
                    knowledge cannot supply the answer
  3. verifiable     the generator knows the answer, so "did the agent use the
                    tool" is checkable rather than assumed
  4. multi-tool     one instance needs >= 2 distinct tools to complete

Acceptance gate
---------------
A task is grounded iff the model CANNOT answer it with the tool registry
removed. `TaskInstance.answerable_without_tools()` states what that check has
to compare; the recorder runs the same prompt with tools=[] and asserts the
ground truth is not reproduced. Any task that survives tools=[] is
knowledge-answerable and must be regenerated.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

# Vocabulary for minted facts. Deliberately mundane: the point is that the
# *pairing* of id -> value is random, not that the words are exotic.
_METRICS = ("p99 latency", "uop-cache coverage", "indirect mispredict rate",
            "L1i MPKI", "retire slot waste", "front-end stall share")
_UNITS = {"p99 latency": "ms", "uop-cache coverage": "%",
          "indirect mispredict rate": "/K", "L1i MPKI": "",
          "retire slot waste": "%", "front-end stall share": "%"}
_CONFIGS = ("baseline", "wide-decode", "large-dsb", "perfect-bp", "warm-pool")
_QUALIFIERS = ("under sustained load", "after a cold start",
               "with prefetch disabled", "at steady state")


@dataclass(frozen=True)
class Fact:
    doc_id: str
    metric: str
    config: str
    value: str
    qualifier: str

    def as_document(self) -> str:
        return (f"[{self.doc_id}] Measurement report. Configuration "
                f"'{self.config}' was evaluated {self.qualifier}. The "
                f"observed {self.metric} was {self.value}. This figure is "
                f"recorded for downstream comparison and should be cited by "
                f"document id.")


@dataclass(frozen=True)
class Term:
    term_id: str
    phrase: str
    definition: str

    def as_entry(self) -> str:
        return f"{self.term_id} ({self.phrase}): {self.definition}"


@dataclass
class TaskInstance:
    """One round's work. Requires a document lookup AND a glossary lookup."""
    round_index: int
    fact: Fact
    term: Term
    tools_required: tuple[str, ...]
    ground_truth: dict = field(default_factory=dict)

    def prompt(self) -> str:
        return (
            f"Round {self.round_index}. Two lookups, both against the indexed "
            f"corpus and glossary you have tools for.\n"
            f"  (1) Document {self.fact.doc_id} records a value for "
            f"'{self.fact.metric}' under configuration "
            f"'{self.fact.config}'. Report that value exactly as written.\n"
            f"  (2) Glossary term {self.term.term_id} has a definition in the "
            f"glossary. Report its defined phrase.\n"
            f"Return JSON: "
            f'{{"doc_value": "...", "term_phrase": "...", "evidence": '
            f'["{self.fact.doc_id}", "{self.term.term_id}"]}}. '
            f"Do not guess: both identifiers were minted for this run and "
            f"appear nowhere else."
        )

    def answerable_without_tools(self, response_text: str) -> bool:
        """Gate helper: True means the task FAILED to be grounded.

        The identifiers and values are minted per run, so reproducing either
        ground-truth string without a tool call means the task leaked the
        answer into the prompt, or the value was guessable.
        """
        return (self.ground_truth["doc_value"] in response_text
                or self.ground_truth["term_phrase"] in response_text)


class GroundedCorpus:
    """Minted corpus + glossary, and the task instances that reference them."""

    def __init__(self, seed: int, n_docs: int = 200, n_terms: int = 64):
        # Derive an independent stream so callers' rng state is untouched and
        # the same (seed, n_docs, n_terms) always mints the same corpus.
        h = hashlib.sha256(f"grounded/{seed}/{n_docs}/{n_terms}".encode()).digest()
        self.rng = random.Random(int.from_bytes(h[:8], "big"))
        self.seed = seed
        # Ground-truth strings must be UNIQUE across the corpus, or
        # "the agent reported the right value" stops proving "the agent read
        # the right document" -- two docs sharing a value would let a wrong
        # lookup pass verification. Measured without this: 2 collisions in
        # 200 docs at seed 1234.
        self.facts = self._mint_unique(n_docs, self._mint_fact, lambda f: f.value)
        self.terms = self._mint_unique(n_terms, self._mint_term, lambda t: t.phrase)

    @staticmethod
    def _mint_unique(n, mint, key):
        out, seen = [], set()
        for i in range(n):
            for _ in range(64):          # bounded: values are 2dp over a wide range
                item = mint(i)
                if key(item) not in seen:
                    break
            else:
                raise RuntimeError(f"could not mint a unique value for index {i}")
            seen.add(key(item))
            out.append(item)
        return out

    def _mint_fact(self, i: int) -> Fact:
        metric = self.rng.choice(_METRICS)
        unit = _UNITS[metric]
        # Value is a random decimal: not derivable, not memorable.
        value = f"{self.rng.uniform(0.5, 99.5):.2f}{unit}"
        return Fact(doc_id=f"DOC-{4000 + i * 7 % 9000:04d}",
                    metric=metric,
                    config=self.rng.choice(_CONFIGS),
                    value=value,
                    qualifier=self.rng.choice(_QUALIFIERS))

    def _mint_term(self, i: int) -> Term:
        # Phrase is a random two-token handle, so it cannot be inferred.
        a = self.rng.randrange(0x1000, 0xFFFF)
        b = self.rng.randrange(0x1000, 0xFFFF)
        phrase = f"{a:04x}-{b:04x}"
        return Term(term_id=f"T-{200 + i:03d}", phrase=phrase,
                    definition=(f"Internal handle {phrase}; assigned during "
                                f"corpus minting for run seed {self.seed}."))

    # ---- what the runner needs -------------------------------------------

    def documents(self) -> list[str]:
        return [f.as_document() for f in self.facts]

    def glossary(self) -> dict[str, str]:
        return {t.term_id: t.as_entry() for t in self.terms}

    def task(self, round_index: int) -> TaskInstance:
        """Deterministic per-round instance."""
        r = random.Random(
            int.from_bytes(
                hashlib.sha256(f"task/{self.seed}/{round_index}".encode()).digest()[:8],
                "big"))
        fact = r.choice(self.facts)
        term = r.choice(self.terms)
        return TaskInstance(
            round_index=round_index,
            fact=fact,
            term=term,
            tools_required=("search_and_summarize", "lookup_definition"),
            ground_truth={"doc_value": fact.value,
                          "term_phrase": term.phrase,
                          "doc_id": fact.doc_id,
                          "term_id": term.term_id},
        )


def selftest(seed: int = 42, rounds: int = 15) -> str:
    """Properties the gate depends on, checked mechanically."""
    c = GroundedCorpus(seed)
    tasks = [c.task(i) for i in range(rounds)]
    lines = [f"GroundedCorpus(seed={seed}): {len(c.facts)} docs, {len(c.terms)} terms", ""]

    # 1. per-round: prompts differ
    distinct = len({t.prompt() for t in tasks})
    lines.append(f"  per-round instances distinct : {distinct}/{rounds}")

    # 2. unknowable: the answer never appears in the prompt
    leaked = sum(1 for t in tasks
                 if t.ground_truth["doc_value"] in t.prompt()
                 or t.ground_truth["term_phrase"] in t.prompt())
    lines.append(f"  prompts leaking ground truth : {leaked}/{rounds}  (must be 0)")

    # 3. multi-tool
    multi = sum(1 for t in tasks if len(t.tools_required) >= 2)
    lines.append(f"  instances needing >=2 tools  : {multi}/{rounds}")

    # 4. determinism: same seed -> same tasks
    c2 = GroundedCorpus(seed)
    same = all(c2.task(i).prompt() == tasks[i].prompt() for i in range(rounds))
    lines.append(f"  deterministic across rebuild : {same}")

    # 5. the answer really is only in the corpus
    docs, gloss = "\n".join(c.documents()), "\n".join(c.glossary().values())
    found = sum(1 for t in tasks
                if t.ground_truth["doc_value"] in docs
                and t.ground_truth["term_phrase"] in gloss)
    lines.append(f"  answers present in corpus    : {found}/{rounds}")

    lines += ["", "  sample prompt (round 0):", "  " + tasks[0].prompt().replace("\n", "\n  ")]
    lines += ["", f"  ground truth: {tasks[0].ground_truth}"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(selftest())
