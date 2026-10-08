"""Suggest catalog articles (with negotiated prices) for free-text order lines.

The customer's complaint: "people raise requests and go off buying whatever
they find" even though specific prices were negotiated for specific articles.
When a request is created, each order line is matched against the
organization's own article catalog and the matches are offered inline.

Matching is lexical and runs in Python over the org's catalog (a few thousand
rows): tokens are lower-cased, lightly stemmed and weighted by inverse document
frequency, so "moss" counts for more than "in". The score blends how much of
the *query* is covered by the article and how much of the *article* is covered
by the query, which keeps "Toner cartridge black" from matching every article
that merely contains "black". Good enough for catalogs of this size; for very
large catalogs the same interface could sit on pg_trgm or an embedding index.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.models import Article

# Words that carry no meaning for matching a product description.
STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "with", "to", "in", "on", "per", "x",
    "incl", "including", "inc", "llc", "ltd", "corp", "co", "approx", "ea",
    "pcs", "pc", "piece", "pieces", "item", "items", "qty", "unit", "units",
}

MIN_SCORE = 0.35
DEFAULT_LIMIT = 3

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _stem(token: str) -> str:
    """Very light stemming: plurals and a couple of common suffixes."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> set[str]:
    tokens = set()
    for raw in _TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS or (len(raw) < 2 and not raw.isdigit()):
            continue
        tokens.add(_stem(raw))
    return tokens


@dataclass
class ArticleMatch:
    article: Article
    score: float
    matched_terms: list[str]


class CatalogIndex:
    """Token index over one organization's articles."""

    def __init__(self, articles: list[Article]):
        self.articles = articles
        self._tokens = [tokenize(a.description) for a in articles]
        df: Counter[str] = Counter()
        for toks in self._tokens:
            df.update(toks)
        n = max(len(articles), 1)
        # Smoothed IDF; a term present in every article still gets a small weight.
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def _weight(self, tokens: set[str]) -> float:
        return sum(self._idf.get(t, 1.0) for t in tokens)

    def suggest(self, query: str, limit: int = DEFAULT_LIMIT) -> list[ArticleMatch]:
        q = tokenize(query)
        # Words the catalog has never seen (model numbers, dimensions) cannot
        # match anything, so they do not count against coverage.
        known = {t for t in q if t in self._idf}
        if not known:
            return []
        # A query of several words must share at least two of them with an
        # article; otherwise "cable ties" would surface every charging cable.
        min_common = min(2, len(q))
        q_weight = self._weight(known)
        matches: list[ArticleMatch] = []
        for article, toks in zip(self.articles, self._tokens):
            common = known & toks
            if len(common) < min_common:
                continue
            common_weight = self._weight(common)
            query_coverage = common_weight / q_weight
            article_coverage = common_weight / self._weight(toks)
            score = 0.6 * query_coverage + 0.4 * article_coverage
            if score >= MIN_SCORE:
                matches.append(ArticleMatch(article, round(score, 3), sorted(common)))
        # Best match first; among equal matches, the cheapest negotiated price.
        matches.sort(key=lambda m: (-m.score, float(m.article.unit_price), m.article.article_number))
        return matches[:limit]
