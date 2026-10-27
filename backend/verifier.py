"""Relevance, claim-support and conflict scoring.

Three back-ends, chosen by VERIFIER in .env:
  nli     - cross-encoder NLI model (entail / contradict probabilities). Best quality.
  llm     - asks the LLM to judge each claim. Slower and costs tokens.
  lexical - term and number overlap. No downloads; good enough for demos and unit tests.
  auto    - nli if the model loads, otherwise lexical.
Relevance uses the cross-encoder reranker when available, else query-term coverage.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np

from . import models
from .config import settings
from .llm import LLM
from .textutils import content_terms, coverage, numbers

_UNIT_NUM = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg/kg|mg|mcg|g|ml|mmol/l|mg/dl|mmhg|minutes?|mins?|hours?|days?|weeks?|%)", re.I
)


_NEG_WORD = re.compile(r"(no|not|never|without|avoid|cannot|contraindicated|absent|none|neither|nor)|.*n't|.*n\u2019t", re.I)


def _neg_contexts(text: str) -> list[set[str]]:
    """For every negation word, the content terms it applies to (up to 4 words, stopping at the
    next negation). "should not be given without delay" gives [{given}, {delay}]."""
    toks = re.findall(r"[A-Za-z0-9'\u2019-]+", text.lower())
    out: list[set[str]] = []
    for i, t in enumerate(toks):
        if _NEG_WORD.fullmatch(t):
            window: list[str] = []
            for u in toks[i + 1 : i + 5]:
                if _NEG_WORD.fullmatch(u):
                    break
                window.append(u)
            out.append(set(content_terms(" ".join(window))))
    return out


def _negation_consistent(claim: str, premise: str) -> bool:
    """False when the claim negates something the premise does not, or the premise negates
    something the claim talks about but the claim does not. A plain count of negation words is
    not enough: two sentences can contain the same number of unrelated negations."""
    c, p = _neg_contexts(claim), _neg_contexts(premise)
    claim_terms = set(content_terms(claim))
    for cc in c:
        if not any(cc & pp for pp in p):
            return False
    for pp in p:
        if pp & claim_terms and not any(pp & cc for cc in c):
            return False
    return True


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _unit_key(unit: str) -> str:
    unit = unit.lower()
    for root in ("minute", "min", "hour", "day", "week"):
        if unit.startswith(root):
            return "minute" if root == "min" else root
    return unit


class Verifier:
    def __init__(self, llm: LLM | None = None):
        self.llm = llm
        self.idf: dict[str, float] = {}  # set by the pipeline from the BM25 index
        self.mode = self._choose_mode()

    def _choose_mode(self) -> str:
        wanted = settings.verifier.lower()
        if wanted in {"auto", "nli"} and models.nli() is not None:
            return "nli"
        if wanted == "llm" and self.llm is not None and not self.llm.is_mock:
            return "llm"
        return "lexical"

    # -------------------------------------------------------------- relevance
    def relevance(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        rr = models.reranker()
        if rr is not None:
            raw = np.asarray(rr.predict([(query, t) for t in texts]), dtype=float)
            if raw.min() < 0 or raw.max() > 1:
                raw = _sigmoid(raw)
            return raw.tolist()
        q = content_terms(query)
        return [coverage(q, t, self.idf) for t in texts]

    # ---------------------------------------------------------------- support
    def _nli_probs(self, pairs: list[tuple[str, str]]) -> list[dict]:
        model = models.nli()
        logits = np.asarray(model.predict(pairs), dtype=float)
        logits = logits.reshape(len(pairs), -1)
        # Some sentence-transformers versions already return probabilities. Softmaxing those
        # again squashes everything towards 1/3 and entailment can never reach the threshold.
        if logits.min() >= 0 and logits.max() <= 1 and np.allclose(logits.sum(axis=1), 1.0, atol=1e-3):
            probs = logits
        else:
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
        id2label = getattr(getattr(model, "config", None), "id2label", None) or {
            0: "contradiction", 1: "entailment", 2: "neutral"
        }
        idx = {v.lower(): int(k) for k, v in id2label.items()}
        return [
            {"entail": float(p[idx.get("entailment", 1)]), "contradict": float(p[idx.get("contradiction", 0)])}
            for p in probs
        ]

    def _lexical_support(self, claim: str, passage: str) -> dict:
        terms = content_terms(claim)
        score = coverage(terms, passage)
        claim_nums = numbers(claim)
        if claim_nums:
            missing = claim_nums - numbers(passage)
            if missing:
                score *= 0.5 ** len(missing)
        contradict = self.number_conflict(claim, passage)
        return {"entail": score, "contradict": contradict}

    @lru_cache(maxsize=4096)
    def _llm_support(self, claim: str, passage: str) -> tuple[float, float]:
        reply = self.llm.chat(
            "You check whether a passage supports a claim. Reply with one word: "
            "SUPPORTED, CONTRADICTED or NOT_ENOUGH_INFO.",
            f"Passage:\n{passage}\n\nClaim:\n{claim}",
            max_tokens=5,
        ).upper()
        if "CONTRADICT" in reply:
            return 0.0, 1.0
        if "SUPPORT" in reply:
            return 1.0, 0.0
        return 0.0, 0.0

    def support(self, claim: str, passages: list[str]) -> list[dict]:
        """For one claim, entail/contradict scores against each passage."""
        if not passages:
            return []
        if self.mode == "nli":
            probs = self._nli_probs([(p, claim) for p in passages])
            # Hybrid: a real LLM paraphrases, and a small NLI model often calls that "neutral".
            # Lexical evidence may lift a "neutral" verdict, but only when it cannot be a bag-of-words
            # accident: (1) the premise is a short window, not a whole 100-word chunk, (2) NLI does
            # not see a contradiction, (3) claim and premise negate the same things, and (4) ALL claim
            # terms (one substituted word, e.g. a wrong route or drug, must not pass) AND every number appear in the premise (lexical_support halves the score
            # per missing number).
            for pr, p in zip(probs, passages):
                if len(p.split()) > settings.lexical_max_words:
                    continue
                if pr["contradict"] >= settings.lexical_max_contradiction:
                    continue
                if not _negation_consistent(claim, p):
                    continue
                lex = self._lexical_support(claim, p)
                if lex["entail"] >= settings.lexical_accept and lex["contradict"] < 0.5:
                    pr["entail"] = max(pr["entail"], lex["entail"])
            return probs
        if self.mode == "llm":
            return [dict(zip(("entail", "contradict"), self._llm_support(claim, p))) for p in passages]
        return [self._lexical_support(claim, p) for p in passages]

    # --------------------------------------------------------------- conflict
    @staticmethod
    def number_conflict(a: str, b: str) -> float:
        """Heuristic: same unit and shared context words, but different quantities."""
        def mentions(text):
            out = []
            for m in _UNIT_NUM.finditer(text):
                window = text[max(0, m.start() - 80) : m.end() + 40]
                out.append((_unit_key(m.group(2)), m.group(1), set(content_terms(window))))
            return out

        ma, mb = mentions(a), mentions(b)
        exact_b = {(u, n) for u, n, _ in mb}
        best = 0.0
        for unit_a, num_a, ctx_a in ma:
            if (unit_a, num_a) in exact_b:
                continue  # the same quantity appears in both texts
            for unit_b, num_b, ctx_b in mb:
                if unit_a == unit_b and num_a != num_b:
                    shared = len(ctx_a & ctx_b)
                    if shared >= 3:
                        best = max(best, min(1.0, 0.5 + 0.1 * shared))
        return best

    def conflict(self, a: str, b: str) -> float:
        """0-1 score that two passages contradict each other."""
        if self.mode == "nli":
            p1, p2 = self._nli_probs([(a, b), (b, a)])
            return max(p1["contradict"], p2["contradict"])
        return self.number_conflict(a, b)


def safe_mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return float(sum(vals) / len(vals)) if vals else 0.0