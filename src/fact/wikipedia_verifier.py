from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class FactCheckResult:
    claim: str
    confidence: float
    matched_title: str | None
    summary: str | None


class WikipediaFactVerifier:
    """
    Minimal external verification hook.

    Uses Wikipedia's public REST summary endpoint for quick weak supervision.
    """

    base_url = "https://en.wikipedia.org/api/rest_v1/page/summary"

    def extract_claims(self, text: str) -> list[str]:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        return sentences[:3]

    def verify_claim(self, claim: str) -> FactCheckResult:
        title_guess = claim.split()[:4]
        if not title_guess:
            return FactCheckResult(claim=claim, confidence=0.0, matched_title=None, summary=None)

        title = "_".join(title_guess)
        try:
            response = requests.get(f"{self.base_url}/{title}", timeout=5)
            if response.status_code != 200:
                return FactCheckResult(claim=claim, confidence=0.0, matched_title=None, summary=None)
            payload: dict[str, Any] = response.json()
            summary = payload.get("extract")
            matched_title = payload.get("title")
            confidence = 0.7 if summary else 0.2
            return FactCheckResult(
                claim=claim,
                confidence=confidence,
                matched_title=matched_title,
                summary=summary,
            )
        except requests.RequestException:
            return FactCheckResult(claim=claim, confidence=0.0, matched_title=None, summary=None)

    def fact_consistency_score(self, text: str) -> float:
        claims = self.extract_claims(text)
        if not claims:
            return 0.0
        checks = [self.verify_claim(c).confidence for c in claims]
        return sum(checks) / len(checks)
