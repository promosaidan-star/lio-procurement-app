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
from __future__ import annotations  # lets type hints reference classes defined later in the file

import math  # log() for the IDF weights
import re  # tokeniser regex
from collections import Counter  # document-frequency counts per token
from dataclasses import dataclass  # small result record without boilerplate

from app.models import Article  # ORM row type; only its attributes are read here

# Words that carry no meaning for matching a product description.
STOPWORDS = {
    "a", "an", "and", "the", "of", "for", "with", "to", "in", "on", "per", "x",  # grammar words and the "63 x 31" separator
    "incl", "including", "inc", "llc", "ltd", "corp", "co", "approx", "ea",  # legal suffixes and quote jargon
    "pcs", "pc", "piece", "pieces", "item", "items", "qty", "unit", "units",  # quantity words
}

MIN_SCORE = 0.35  # matches below this are not worth showing
DEFAULT_LIMIT = 3  # suggestions per order line unless the caller asks for more

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")  # words and numbers, keeping decimals like 31.5 whole


def _stem(token: str) -> str:
    """Very light stemming: plurals and a couple of common suffixes."""
    if len(token) > 4 and token.endswith("ies"):  # "batteries" -> "battery"
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "xes", "zes", "ches", "shes")):  # "boxes" -> "box"
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):  # "cartridges" -> "cartridge", but "glass" stays
        return token[:-1]
    return token  # anything else is left alone


def tokenize(text: str) -> set[str]:
    tokens = set()  # a set: word order and repeats do not matter for matching
    for raw in _TOKEN_RE.findall(text.lower()):  # case-insensitive
        if raw in STOPWORDS or (len(raw) < 2 and not raw.isdigit()):  # drop noise and single letters (keep "2")
            continue
        tokens.add(_stem(raw))  # store the stemmed form
    return tokens


@dataclass
class ArticleMatch:
    article: Article  # the catalog row
    score: float  # 0..1, higher is better
    matched_terms: list[str]  # which words of the query the article shares (shown for transparency)


class CatalogIndex:
    """Token index over one organization's articles."""

    def __init__(self, articles: list[Article]):
        self.articles = articles  # keep the rows so matches can return them
        self._tokens = [tokenize(a.description) for a in articles]  # token set per article, same order
        df: Counter[str] = Counter()  # document frequency: how many articles contain each token
        for toks in self._tokens:
            df.update(toks)  # a token counts once per article because toks is a set
        n = max(len(articles), 1)  # avoid division by zero on an empty catalog
        # Smoothed IDF; a term present in every article still gets a small weight.
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def _weight(self, tokens: set[str]) -> float:
        return sum(self._idf.get(t, 1.0) for t in tokens)  # total rarity weight of a token set

    def suggest(self, query: str, limit: int = DEFAULT_LIMIT) -> list[ArticleMatch]:
        q = tokenize(query)  # tokens of the order-line description
        # Words the catalog has never seen (model numbers, dimensions) cannot
        # match anything, so they do not count against coverage.
        known = {t for t in q if t in self._idf}
        if not known:  # nothing in the query exists in the catalog vocabulary
            return []
        # A query of several words must share at least two of them with an
        # article; otherwise "cable ties" would surface every charging cable.
        min_common = min(2, len(q))
        q_weight = self._weight(known)  # denominator for query coverage
        matches: list[ArticleMatch] = []  # candidates above the threshold
        for article, toks in zip(self.articles, self._tokens):  # scan every article once
            common = known & toks  # tokens shared by query and article
            if len(common) < min_common:  # not enough shared words
                continue
            common_weight = self._weight(common)  # rarity-weighted overlap
            query_coverage = common_weight / q_weight  # how much of the query the article explains
            article_coverage = common_weight / self._weight(toks)  # how much of the article the query explains
            score = 0.6 * query_coverage + 0.4 * article_coverage  # query side matters a bit more
            if score >= MIN_SCORE:  # keep only plausible matches
                matches.append(ArticleMatch(article, round(score, 3), sorted(common)))
        # Best match first; among equal matches, the cheapest negotiated price.
        matches.sort(key=lambda m: (-m.score, float(m.article.unit_price), m.article.article_number))
        return matches[:limit]  # top N only
