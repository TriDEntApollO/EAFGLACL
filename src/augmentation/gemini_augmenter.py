from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types


class GeminiAugmentationError(RuntimeError):
    """Raised when Gemini augmentation fails for a source record."""


@dataclass
class GeminiAugmentationClient:
    """Thin Gemini client wrapper for controllable text augmentation."""

    api_key: str
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.4
    top_p: float = 0.95
    max_output_tokens: int = 512
    prompt_version: str = "v1"
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = genai.Client(api_key=self.api_key)

    def validate_preflight(self, project_name: str | None, project_number: str | None) -> None:
        if not self.api_key:
            raise GeminiAugmentationError("GEMINI_API_KEY is empty or missing.")

        normalized_name = (project_name or "").strip()
        normalized_number = (project_number or "").strip()
        if not normalized_name or not normalized_number:
            raise GeminiAugmentationError(
                "Missing project configuration. Set PROJECT_NAME and PROJECT_NUMBER in .env."
            )

        expected_name = f"projects/{normalized_number}"
        if normalized_name != expected_name:
            raise GeminiAugmentationError(
                f"Project mismatch: PROJECT_NAME={normalized_name!r} does not match PROJECT_NUMBER={normalized_number!r}. "
                f"Expected PROJECT_NAME={expected_name!r}."
            )

        try:
            model_entries = list(self._client.models.list())
        except Exception as exc:  # pragma: no cover - third-party SDK exceptions vary
            raise GeminiAugmentationError(f"Preflight failed to list Gemini models: {exc}") from exc

        available_names: set[str] = set()
        for model in model_entries:
            name = str(getattr(model, "name", "") or "").strip()
            if name:
                available_names.add(name)
                if name.startswith("models/"):
                    available_names.add(name.split("/", 1)[1])

        if self.model_name not in available_names and f"models/{self.model_name}" not in available_names:
            sample = sorted(n for n in available_names if n.startswith("gemini-") or n.startswith("models/gemini-"))
            preview = ", ".join(sample[:10]) if sample else "<none>"
            raise GeminiAugmentationError(
                f"Configured model '{self.model_name}' is not available for this key/project. Available models: {preview}"
            )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents="Return exactly: OK",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    top_p=1.0,
                    max_output_tokens=8,
                ),
            )
            _ = response
        except GeminiAugmentationError:
            raise
        except Exception as exc:  # pragma: no cover - third-party SDK exceptions vary
            raise GeminiAugmentationError(f"Preflight generation failed for model '{self.model_name}': {exc}") from exc

    def fetch_rate_limits(self) -> dict[str, int | None]:
        limits: dict[str, int | None] = {"rpm": None, "tpm": None, "rpd": None}

        # Fetch model metadata first (input token limit is the closest API-exposed TPM signal).
        try:
            model_info = self._client.models.get(model=self.model_name)
            input_token_limit = getattr(model_info, "input_token_limit", None)
            if isinstance(input_token_limit, int) and input_token_limit > 0:
                limits["tpm"] = input_token_limit
        except Exception:
            pass

        # Probe generate_content once and extract any quota values exposed by API errors.
        try:
            self._client.models.generate_content(
                model=self.model_name,
                contents="OK",
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=1),
            )
        except Exception as exc:  # pragma: no cover - third-party SDK exceptions vary
            extracted = self._extract_quota_values(exc)
            for key in ("rpm", "tpm", "rpd"):
                if extracted.get(key) is not None:
                    limits[key] = extracted[key]

        return limits

    def _extract_quota_values(self, exc: Exception) -> dict[str, int | None]:
        limits: dict[str, int | None] = {"rpm": None, "tpm": None, "rpd": None}
        response_json = getattr(exc, "response_json", None)
        if not isinstance(response_json, dict):
            return limits

        details = response_json.get("error", {}).get("details", [])
        if not isinstance(details, list):
            return limits

        for detail in details:
            if not isinstance(detail, dict):
                continue
            if not str(detail.get("@type", "")).endswith("QuotaFailure"):
                continue
            violations = detail.get("violations", [])
            if not isinstance(violations, list):
                continue
            for violation in violations:
                if not isinstance(violation, dict):
                    continue
                quota_id = str(violation.get("quotaId", ""))
                metric = str(violation.get("quotaMetric", "")).lower()
                quota_value = violation.get("quotaValue")
                try:
                    value = int(quota_value) if quota_value is not None else None
                except (TypeError, ValueError):
                    value = None
                if value is None:
                    continue

                if "PerDay" in quota_id:
                    limits["rpd"] = max(limits["rpd"] or 0, value)
                elif "PerMinute" in quota_id and "request" in metric:
                    limits["rpm"] = max(limits["rpm"] or 0, value)
                elif "PerMinute" in quota_id and "token" in metric:
                    limits["tpm"] = max(limits["tpm"] or 0, value)

        return limits

    def generate_candidates(
        self,
        source_text: str,
        label: str,
        dataset: str,
        candidate_count: int,
    ) -> list[str]:
        if candidate_count < 1:
            return []

        prompt = self._build_prompt(
            source_text=source_text,
            label=label,
            dataset=dataset,
            candidate_count=candidate_count,
        )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_output_tokens,
                    response_mime_type="application/json",
                ),
            )
        except Exception as exc:  # pragma: no cover - third-party SDK exceptions vary
            raise GeminiAugmentationError(f"Gemini API call failed: {exc}") from exc

        raw = self._extract_response_text(response)
        if not raw:
            raise GeminiAugmentationError("Gemini returned empty response text")

        parsed = self._parse_candidates(raw)
        if not parsed:
            raise GeminiAugmentationError("Gemini response contained no usable candidates")

        return parsed[:candidate_count]

    def generate_candidates_batch(
        self,
        records: list[dict[str, str]],
        candidate_count: int,
    ) -> dict[str, list[str]]:
        if not records:
            return {}
        if candidate_count < 1:
            return {r["record_id"]: [] for r in records}

        prompt = self._build_batch_prompt(records=records, candidate_count=candidate_count)

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_output_tokens,
                    response_mime_type="application/json",
                ),
            )
        except Exception as exc:  # pragma: no cover - third-party SDK exceptions vary
            raise GeminiAugmentationError(f"Gemini batch API call failed: {exc}") from exc

        parsed_payload = self._extract_response_payload(response)
        if parsed_payload is not None:
            parsed_from_payload = self._parse_batch_payload(parsed_payload)
            if parsed_from_payload:
                return parsed_from_payload

        raw = self._extract_response_text(response)
        if not raw:
            raise GeminiAugmentationError("Gemini batch response text was empty")

        parsed = self._parse_batch_candidates(raw)
        if not parsed:
            raise GeminiAugmentationError("Gemini batch response contained no usable candidates")
        return parsed

    def _parse_batch_payload(self, payload: Any) -> dict[str, list[str]]:
        if not isinstance(payload, dict):
            return {}

        items: list[Any]
        maybe_items = payload.get("items", [])
        if isinstance(maybe_items, list):
            items = maybe_items
        elif maybe_items:
            items = [maybe_items]
        else:
            obj_results: dict[str, list[str]] = {}
            for key, value in payload.items():
                if not isinstance(value, list):
                    continue
                cleaned = [c.strip() for c in value if isinstance(c, str) and c.strip()]
                if cleaned:
                    obj_results[str(key).strip()] = cleaned
            return obj_results

        results: dict[str, list[str]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            record_id = str(
                item.get("record_id", item.get("recordId", item.get("id", "")))
            ).strip()
            candidates = item.get(
                "candidates",
                item.get("rewrites", item.get("variations", item.get("outputs", []))),
            )
            if not record_id or not isinstance(candidates, list):
                continue
            cleaned = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
            if cleaned:
                results[record_id] = cleaned
        return results

    def _build_prompt(self, source_text: str, label: str, dataset: str, candidate_count: int) -> str:
        return (
            "You are generating high-quality misinformation-detection data augmentations.\n"
            "Task: produce semantically consistent rewrites that preserve emotional tone and stance.\n"
            "Rules:\n"
            "1) Keep the same claim intent and rumor stance.\n"
            "2) Preserve emotional tone (fear/anger/sarcasm/neutral).\n"
            "3) Do not invent new entities or factual claims.\n"
            "4) Keep each candidate as a single sentence or short paragraph.\n"
            f"Return EXACT JSON: {{\"candidates\": [\"...\"]}} with {candidate_count} strings.\n"
            f"Dataset: {dataset}\n"
            f"Label: {label}\n"
            f"Source text:\n{source_text}"
        )

    def _build_batch_prompt(self, records: list[dict[str, str]], candidate_count: int) -> str:
        serialized_records = json.dumps(records, ensure_ascii=False)
        return (
            "You are generating high-quality misinformation-detection data augmentations.\n"
            "For EACH input item, create semantically consistent rewrites that preserve emotion and stance.\n"
            "Rules:\n"
            "1) Keep claim intent and rumor stance unchanged.\n"
            "2) Preserve emotional tone.\n"
            "3) Do not introduce new entities or facts.\n"
            "4) Keep each candidate concise and natural.\n"
            f"Return EXACT JSON object with key \"items\". Each item must be {{\"record_id\": string, \"candidates\": string[]}} and include {candidate_count} candidates.\n"
            "No markdown, no explanation text.\n"
            f"Input items JSON:\n{serialized_records}"
        )

    def _parse_candidates(self, raw_text: str) -> list[str]:
        raw_text = raw_text.strip()

        # Primary path: strict JSON output as requested.
        try:
            payload = json.loads(raw_text)
            candidates = payload.get("candidates", [])
            return [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
        except json.JSONDecodeError:
            pass

        # Fallback path: tolerate markdown fenced JSON.
        if raw_text.startswith("```"):
            stripped = raw_text.strip("`")
            stripped = stripped.replace("json\n", "", 1)
            try:
                payload = json.loads(stripped)
                candidates = payload.get("candidates", [])
                return [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
            except json.JSONDecodeError:
                pass

        # Last resort: parse line-based bullets/numbering.
        parsed: list[str] = []
        for line in raw_text.splitlines():
            line = line.strip().lstrip("-*")
            if not line:
                continue
            if ". " in line and line.split(". ", 1)[0].isdigit():
                line = line.split(". ", 1)[1]
            if line:
                parsed.append(line.strip())
        return parsed

    def _parse_batch_candidates(self, raw_text: str) -> dict[str, list[str]]:
        payload = self._parse_json_payload(raw_text)
        if payload is None:
            return {}
        return self._parse_batch_payload(payload)

    def _parse_json_payload(self, raw_text: str) -> dict[str, Any] | None:
        raw_text = raw_text.strip()
        try:
            payload = json.loads(raw_text)
            if isinstance(payload, dict):
                return payload
            return None
        except json.JSONDecodeError:
            pass

        # Recover embedded JSON object from noisy wrappers.
        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = raw_text[first_brace : last_brace + 1]
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                try:
                    payload = ast.literal_eval(candidate)
                    if isinstance(payload, dict):
                        return payload
                except (ValueError, SyntaxError):
                    pass

        if raw_text.startswith("```"):
            stripped = raw_text.strip("`")
            stripped = stripped.replace("json\n", "", 1)
            try:
                payload = json.loads(stripped)
                return payload if isinstance(payload, dict) else None
            except json.JSONDecodeError:
                try:
                    payload = ast.literal_eval(stripped)
                    return payload if isinstance(payload, dict) else None
                except (ValueError, SyntaxError):
                    return None
        return None

    def _extract_response_text(self, response: Any) -> str:
        direct = (getattr(response, "text", "") or "").strip()
        if direct:
            return direct

        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""

        chunks: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if not parts:
                continue
            for part in parts:
                part_text = (getattr(part, "text", "") or "").strip()
                if part_text:
                    chunks.append(part_text)

        return "\n".join(chunks).strip()

    def _extract_response_payload(self, response: Any) -> dict[str, Any] | None:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        if parsed is not None:
            try:
                maybe_dict = json.loads(json.dumps(parsed))
                if isinstance(maybe_dict, dict):
                    return maybe_dict
            except Exception:
                pass
        return None
