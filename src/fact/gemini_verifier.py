import os
import ast
import time
import logging
import wikipedia
from google import genai
from google.genai import types
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from .wikipedia_verifier import FactCheckResult  # single source of truth

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash-lite"  # update here if model name changes


# ─────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────
class GeminiFactVerifier:
    """
    Verifies factual claims in text using a two-step approach:
      1. Gemini extracts claims + generates search queries in ONE call
      2. Wikipedia retrieves evidence
      3. Gemini verifies each claim against evidence
    Returns a FactCheckResult compatible with the project pipeline.
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found. "
                "Add it to your .env file:\n  GEMINI_API_KEY=your_key_here"
            )
        self.client = genai.Client(api_key=api_key)

    # ── STEP 1 ────────────────────────────────────────────────
    # Extract claims AND generate search queries in ONE Gemini call
    # (reduces API calls from 2 per claim → 1 per claim)
    # ─────────────────────────────────────────────────────────
    def _extract_claims_with_queries(self, text: str) -> List[Dict[str, str]]:
        """
        Returns a list of dicts:
            [{"claim": "...", "query": "..."}, ...]
        Returns [] if no factual claims exist.
        """
        prompt = f"""Analyze this text and extract all verifiable factual claims.
For each claim, also write a short Wikipedia search query (max 6 words).

Text: "{text}"

Return ONLY a Python list of dicts. No explanation, no markdown, no extra text.
If there are no factual claims, return: []

Format:
[
  {{"claim": "the factual claim", "query": "short search query"}},
  ...
]"""

        try:
            response = self.client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    http_options=types.HttpOptions(timeout=30000),
                ),
            )
            raw = (response.text or "").strip()
            if not raw:
                logger.warning("Gemini returned empty response for claim extraction.")
                return []

            # Strip markdown code fences if Gemini wraps in them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("python"):
                    raw = raw[6:]
            raw = raw.strip()

            parsed = ast.literal_eval(raw)   # safe alternative to eval()

            if isinstance(parsed, list):
                # Validate structure — keep only well-formed entries
                return [
                    item for item in parsed
                    if isinstance(item, dict)
                    and "claim" in item
                    and "query" in item
                ]
            return []

        except Exception as e:
            logger.warning("Claim extraction failed: %s", e)
            return []

    # ── STEP 2 ────────────────────────────────────────────────
    def _retrieve_evidence(self, query: str) -> Tuple[str, str]:
        """
        Searches Wikipedia using the pre-generated query.
        Returns (article_title, article_content_snippet).
        Returns ("", "") if nothing found.
        """
        try:
            results = wikipedia.search(query, results=3)
            if not results:
                return ("", "")

            page = wikipedia.page(results[0], auto_suggest=False)
            # First 1500 chars is enough context; keeps token usage low
            return (page.title, page.content[:1500])

        except wikipedia.exceptions.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return (page.title, page.content[:1500])
            except Exception:
                return ("", "")
        except Exception as e:
            logger.warning("Evidence retrieval failed: %s", e)
            return ("", "")

    # ── STEP 3 ────────────────────────────────────────────────
    def _verify_claim(self, claim: str, evidence: str) -> float:
        """
        Asks Gemini to compare claim vs evidence.
        Returns 1.0 / 0.5 / 0.0.
        """
        if not evidence:
            return 0.5  # No evidence = uncertain, not false

        prompt = f"""You are a strict fact-checker. Based ONLY on the evidence below, classify the claim.

Claim: "{claim}"
Evidence: "{evidence}"

Rules:
- Supported: evidence clearly confirms the claim
- Refuted: evidence clearly contradicts the claim
- Neutral: evidence is unclear, unrelated, or insufficient

Reply with EXACTLY one word: Supported, Refuted, or Neutral"""

        try:
            response = self.client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    http_options=types.HttpOptions(timeout=30000),
                ),
            )
            verdict = (response.text or "").strip().lower()
            if not verdict:
                logger.warning("Gemini returned empty verdict. Defaulting to 0.5")
                return 0.5

            if verdict == "supported":
                return 1.0
            elif verdict == "refuted":
                return 0.0
            elif verdict == "neutral":
                return 0.5
            else:
                logger.warning("Unexpected verdict from Gemini: '%s'. Defaulting to 0.5", verdict)
                return 0.5

        except Exception as e:
            logger.warning("Verification failed: %s", e)
            return 0.5

    # ── PUBLIC METHOD ─────────────────────────────────────────
    def get_fact_score(self, text: str, verbose: bool = False) -> FactCheckResult:
        """
        Full pipeline. Call this from the rest of the project.

        Args:
            text    : Input social media post or sentence
            verbose : Print step-by-step details (for debugging)

        Returns:
            FactCheckResult with confidence score 0.0–1.0
        """
        if verbose:
            print(f"\n[INPUT] {text}")

        # Step 1: Extract claims + queries in one Gemini call
        claims_data = self._extract_claims_with_queries(text)

        if verbose:
            print(f"[CLAIMS] {claims_data}")

        if not claims_data:
            if verbose:
                print("[RESULT] No factual claims. Returning neutral.")
            return FactCheckResult(
                claim=text,
                confidence=0.5,
                matched_title=None,
                summary="No verifiable factual claims found in input."
            )

        scores = []
        last_title = None

        for item in claims_data:
            claim = item["claim"]
            query = item["query"]

            if verbose:
                print(f"\n  → Claim : {claim}")
                print(f"  → Query : {query}")

            # Step 2: Evidence retrieval
            title, evidence = self._retrieve_evidence(query)
            last_title = title or last_title

            if verbose:
                print(f"  → Source: {title or 'Not found'}")
                print(f"  → Evidence (truncated): {evidence[:150]}...")

            # Step 3: Verify
            score = self._verify_claim(claim, evidence)
            scores.append(score)

            if verbose:
                verdict = {1.0: "Supported", 0.5: "Neutral", 0.0: "Refuted"}.get(score, "?")
                print(f"  → Verdict: {verdict} ({score})")

            time.sleep(0.2)  # Minimal guard against free-tier rate limits

        # Average across all claims
        final_score = round(sum(scores) / len(scores), 4)

        if verbose:
            print(f"\n[FINAL SCORE] {final_score}\n")

        return FactCheckResult(
            claim=text,
            confidence=final_score,
            matched_title=last_title,
            summary=f"{len(scores)} claim(s) verified using Gemini + Wikipedia. Confidence: {final_score}"
        )


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST  (run: python gemini_verifier.py)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("True fact",    "The Eiffel Tower is located in Paris, France."),
        ("Misinformation", "5G towers spread the coronavirus."),
        ("Opinion only", "This politician is absolutely terrible."),
        ("False claim",  "COVID vaccines cause infertility in women."),
    ]

    verifier = GeminiFactVerifier()
    print("=" * 60)
    print("GEMINI FACT VERIFIER — TEST RUN")
    print("=" * 60)

    for label, text in test_cases:
        print(f"\n[{label}]")
        result = verifier.get_fact_score(text, verbose=True)
        print(f"confidence    : {result.confidence}")
        print(f"matched_title : {result.matched_title}")
        print(f"summary       : {result.summary}")
        print("-" * 60)