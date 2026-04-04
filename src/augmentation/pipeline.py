from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gemini_augmenter import GeminiAugmentationClient, GeminiAugmentationError


@dataclass
class AugmentationConfig:
    input_path: Path
    output_path: Path
    stats_path: Path
    max_records: int | None = None
    candidates_per_record: int = 3
    model_name: str = "gemini-2.0-flash"
    prompt_version: str = "v1"
    temperature: float = 0.4
    top_p: float = 0.95
    max_output_tokens: int = 512
    target_entries_per_request: int = 100
    rpm_limit: int | None = None
    tpm_limit: int | None = None
    rpd_limit: int | None = None
    rolling_window_seconds: int = 60
    max_retries: int = 5
    backoff_base_seconds: float = 2.0


class RollingRateLimiter:
    """Rolling-window limiter that enforces both RPM and TPM constraints."""

    def __init__(
        self,
        rpm_limit: int | None,
        tpm_limit: int | None,
        rolling_window_seconds: int = 60,
    ) -> None:
        if rpm_limit is not None and rpm_limit < 1:
            raise ValueError("rpm_limit must be >= 1 when provided")
        if tpm_limit is not None and tpm_limit < 1:
            raise ValueError("tpm_limit must be >= 1 when provided")
        if rolling_window_seconds < 1:
            raise ValueError("rolling_window_seconds must be >= 1")

        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rolling_window_seconds = float(rolling_window_seconds)
        self.min_request_spacing = (60.0 / float(rpm_limit)) if rpm_limit else None

        self._request_times: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()
        self._last_request_time: float | None = None

    def _prune(self, now: float) -> None:
        cutoff = now - self.rolling_window_seconds
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()

    def wait_for_capacity(self, tokens_required: int) -> float:
        if tokens_required < 0:
            raise ValueError("tokens_required must be >= 0")

        waited_seconds = 0.0
        epsilon = 0.01

        while True:
            now = time.monotonic()
            self._prune(now)
            waits: list[float] = []

            # Enforce smooth request spacing to avoid bursts.
            if self.min_request_spacing is not None and self._last_request_time is not None:
                spacing_wait = self.min_request_spacing - (now - self._last_request_time)
                if spacing_wait > 0:
                    waits.append(spacing_wait)

            # Rolling-window RPM constraint.
            if self.rpm_limit is not None and len(self._request_times) >= self.rpm_limit:
                waits.append(self.rolling_window_seconds - (now - self._request_times[0]) + epsilon)

            # Rolling-window TPM constraint.
            tokens_used = sum(tokens for _, tokens in self._token_events)
            projected = tokens_used + tokens_required
            if self.tpm_limit is not None and projected > self.tpm_limit:
                excess = projected - self.tpm_limit
                expiring = 0
                for ts, tokens in self._token_events:
                    expiring += tokens
                    if expiring >= excess:
                        waits.append(self.rolling_window_seconds - (now - ts) + epsilon)
                        break

            if not waits:
                return waited_seconds

            sleep_seconds = max(0.0, min(waits))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
                waited_seconds += sleep_seconds

    def record_request(self, tokens_used: int) -> None:
        now = time.monotonic()
        self._prune(now)
        self._request_times.append(now)
        self._token_events.append((now, max(0, tokens_used)))
        self._last_request_time = now

    def snapshot(self) -> dict[str, int | float]:
        now = time.monotonic()
        self._prune(now)
        return {
            "requests_in_window": len(self._request_times),
            "rpm_limit": self.rpm_limit if self.rpm_limit is not None else -1,
            "tokens_in_window": sum(tokens for _, tokens in self._token_events),
            "tpm_limit": self.tpm_limit if self.tpm_limit is not None else -1,
            "window_seconds": self.rolling_window_seconds,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def _extract_retry_delay_seconds(error_text: str) -> float | None:
    # Prefer structured retry info when present, e.g. "retryDelay': '42s'".
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]([0-9]+(?:\.[0-9]+)?)s['\"]", error_text)
    if match:
        return float(match.group(1))

    # Fallback to human-readable text, e.g. "Please retry in 42.326358809s.".
    match = re.search(r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s", error_text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))

    return None


def run_train_augmentation(config: AugmentationConfig) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Set it in your .env file.")

    if not config.input_path.exists():
        raise FileNotFoundError(f"Augmentation input file not found: {config.input_path}")

    source_rows = _read_jsonl(config.input_path)
    if config.max_records is not None:
        source_rows = source_rows[: config.max_records]

    client = GeminiAugmentationClient(
        api_key=api_key,
        model_name=config.model_name,
        temperature=config.temperature,
        top_p=config.top_p,
        max_output_tokens=config.max_output_tokens,
        prompt_version=config.prompt_version,
    )
    project_name = os.getenv("PROJECT_NAME", "").strip()
    project_number = os.getenv("PROJECT_NUMBER", "").strip()
    print(
        f"[augmentation] preflight start: model={config.model_name} project_name={project_name or '<missing>'} "
        f"project_number={project_number or '<missing>'}"
    )

    preflight_error: GeminiAugmentationError | None = None
    preflight_wait_seconds = 0.0
    for attempt in range(config.max_retries):
        try:
            client.validate_preflight(project_name=project_name, project_number=project_number)
            preflight_error = None
            break
        except GeminiAugmentationError as exc:
            preflight_error = exc
            msg_raw = str(exc)
            msg = msg_raw.lower()
            is_rate_limited = "429" in msg or "rate limit" in msg or "quota" in msg

            if not is_rate_limited or attempt >= config.max_retries - 1:
                break

            retry_after = _extract_retry_delay_seconds(msg_raw)
            backoff = retry_after if retry_after is not None else config.backoff_base_seconds * (2**attempt)
            preflight_wait_seconds += backoff
            print(
                f"[augmentation] preflight rate-limited attempt={attempt + 1}/{config.max_retries}; "
                f"retrying in {backoff:.2f}s"
            )
            time.sleep(backoff)

    if preflight_error is not None:
        print(f"[augmentation] preflight failed: {preflight_error}")
        stats = {
            "input_records": len(source_rows),
            "generated_candidates": 0,
            "accepted_pairs": 0,
            "failed_generations": len(source_rows),
            "acceptance_rate": 0.0,
            "model_name": config.model_name,
            "prompt_version": config.prompt_version,
            "candidates_per_record": config.candidates_per_record,
            "target_entries_per_request": config.target_entries_per_request,
            "rpm_limit": config.rpm_limit,
            "tpm_limit": config.tpm_limit,
            "rpd_limit": config.rpd_limit,
            "requested_batches": 0,
            "rate_limit_wait_seconds": round(preflight_wait_seconds, 3),
            "rate_limit_resets": 0,
            "rolling_window_seconds": config.rolling_window_seconds,
            "max_retries": config.max_retries,
            "backoff_base_seconds": config.backoff_base_seconds,
            "input_path": str(config.input_path),
            "output_path": str(config.output_path),
            "preflight_failed": True,
            "preflight_error": str(preflight_error),
        }

        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text("", encoding="utf-8")
        config.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with config.stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
        return stats

    discovered_limits = client.fetch_rate_limits()
    if config.rpm_limit is None:
        config.rpm_limit = discovered_limits.get("rpm")
    if config.tpm_limit is None:
        config.tpm_limit = discovered_limits.get("tpm")
    if config.rpd_limit is None:
        config.rpd_limit = discovered_limits.get("rpd")

    if config.rpm_limit is None:
        print("[augmentation] warning: API did not expose RPM; proceeding without RPM gating")
    if config.tpm_limit is None:
        print("[augmentation] warning: API did not expose TPM; proceeding without TPM gating")

    print(
        f"[augmentation] effective_limits rpm={config.rpm_limit} tpm={config.tpm_limit} "
        f"rpd={config.rpd_limit if config.rpd_limit is not None else 'unknown'}"
    )
    print("[augmentation] preflight passed")

    rate_limiter = RollingRateLimiter(
        rpm_limit=config.rpm_limit,
        tpm_limit=config.tpm_limit,
        rolling_window_seconds=config.rolling_window_seconds,
    )

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_handle = config.output_path.open("w", encoding="utf-8")

    failures = 0
    total_generated = 0
    accepted_pairs = 0
    requested_batches = 0
    rate_limit_wait_seconds = 0.0
    rate_limit_resets = 0
    processed_records = 0
    batch_number = 0

    try:
        index = 0
        while index < len(source_rows):
            batch_number += 1
            remaining = len(source_rows) - index
            batch_size = min(config.target_entries_per_request, remaining)

            # Adaptively reduce batch size until estimated request tokens fit TPM budget.
            est_tokens = _estimate_request_tokens(
                source_rows[index : index + batch_size],
                candidates_per_record=config.candidates_per_record,
            )
            while config.tpm_limit is not None and batch_size > 1 and est_tokens > config.tpm_limit:
                batch_size = max(1, batch_size // 2)
                est_tokens = _estimate_request_tokens(
                    source_rows[index : index + batch_size],
                    candidates_per_record=config.candidates_per_record,
                )

            if config.tpm_limit is not None and est_tokens > config.tpm_limit and batch_size == 1:
                failures += 1
                index += 1
                processed_records += 1
                completion = (processed_records / len(source_rows)) * 100 if source_rows else 100.0
                state = rate_limiter.snapshot()
                print(
                    f"[augmentation] batch={batch_number} skipped_record=1 progress={processed_records}/{len(source_rows)} "
                    f"({completion:.2f}%) requests={state['requests_in_window']}/{state['rpm_limit']} "
                    f"tokens={state['tokens_in_window']}/{state['tpm_limit']}"
                )
                continue

            batch_rows = source_rows[index : index + batch_size]
            request_items: list[dict[str, str]] = []
            for offset, row in enumerate(batch_rows):
                source_text = str(row.get("text", "")).strip()
                if not source_text:
                    continue
                raw_record_id = str(row.get("record_id", "")).strip()
                record_id = raw_record_id or f"row:{index + offset}"
                request_items.append(
                    {
                        "record_id": record_id,
                        "dataset": str(row.get("dataset", "")).strip(),
                        "label": str(row.get("label", "")).strip(),
                        "text": source_text,
                        "source_text": source_text,
                    }
                )

            if not request_items:
                index += batch_size
                processed_records += batch_size
                completion = (processed_records / len(source_rows)) * 100 if source_rows else 100.0
                state = rate_limiter.snapshot()
                print(
                    f"[augmentation] batch={batch_number} empty_request_items progress={processed_records}/{len(source_rows)} "
                    f"({completion:.2f}%) requests={state['requests_in_window']}/{state['rpm_limit']} "
                    f"tokens={state['tokens_in_window']}/{state['tpm_limit']}"
                )
                continue

            batch_generated: dict[str, list[str]] | None = None
            last_error: Exception | None = None

            for attempt in range(config.max_retries):
                if config.rpd_limit is not None and requested_batches >= config.rpd_limit:
                    last_error = GeminiAugmentationError(
                        f"Daily request limit reached (RPD={config.rpd_limit}) for model {config.model_name}"
                    )
                    break

                waited = rate_limiter.wait_for_capacity(est_tokens)
                if waited > 0:
                    rate_limit_wait_seconds += waited
                    rate_limit_resets += 1

                requested_batches += 1
                try:
                    generated = client.generate_candidates_batch(
                        records=request_items,
                        candidate_count=config.candidates_per_record,
                    )
                    rate_limiter.record_request(est_tokens)

                    # Structured output validation: require a mapping entry for each requested record.
                    missing = [item["record_id"] for item in request_items if item["record_id"] not in generated]
                    if missing:
                        raise GeminiAugmentationError(
                            f"Malformed batch response: missing candidates for {len(missing)} records"
                        )

                    batch_generated = generated
                    break
                except GeminiAugmentationError as exc:
                    last_error = exc
                    msg_raw = str(exc)
                    msg = msg_raw.lower()
                    is_rate_limited = "429" in msg or "rate limit" in msg or "quota" in msg

                    if attempt >= config.max_retries - 1:
                        break

                    # Prefer server-provided retry delay for quota errors; else exponential backoff.
                    retry_after = _extract_retry_delay_seconds(msg_raw) if is_rate_limited else None
                    backoff = retry_after if retry_after is not None else config.backoff_base_seconds * (2**attempt)
                    if is_rate_limited:
                        rate_limit_wait_seconds += backoff
                    time.sleep(backoff)

            batch_generated_count = 0
            batch_written_count = 0
            if batch_generated is None:
                failures += len(request_items)
                if last_error is not None:
                    print(f"[augmentation] batch={batch_number} failed: {last_error}")
                index += batch_size
                processed_records += batch_size
                completion = (processed_records / len(source_rows)) * 100 if source_rows else 100.0
                state = rate_limiter.snapshot()
                print(
                    f"[augmentation] batch={batch_number} progress={processed_records}/{len(source_rows)} ({completion:.2f}%) "
                    f"generated=0 appended=0 cumulative_appended={accepted_pairs} "
                    f"requests={state['requests_in_window']}/{state['rpm_limit']} "
                    f"tokens={state['tokens_in_window']}/{state['tpm_limit']}"
                )
                continue

            for item in request_items:
                record_id = item["record_id"]
                generated = batch_generated.get(record_id, [])
                if not generated:
                    failures += 1
                    continue

                total_generated += len(generated)
                batch_generated_count += len(generated)

                source_text = item.get("source_text", "")
                label = item.get("label", "")
                dataset = item.get("dataset", "")

                for candidate in generated:
                    row = {
                        "source_record_id": record_id,
                        "dataset": dataset,
                        "label": label,
                        "source_text": source_text,
                        "candidate_text": candidate,
                        "generation_model": client.model_name,
                        "prompt_version": client.prompt_version,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                    _append_jsonl(output_handle, row)
                    accepted_pairs += 1
                    batch_written_count += 1

            if batch_written_count > 0:
                print(
                    f"[augmentation] batch={batch_number} appended={batch_written_count} rows to {config.output_path} "
                    f"(cumulative={accepted_pairs})"
                )

            index += batch_size
            processed_records += batch_size
            completion = (processed_records / len(source_rows)) * 100 if source_rows else 100.0
            state = rate_limiter.snapshot()
            print(
                f"[augmentation] batch={batch_number} progress={processed_records}/{len(source_rows)} ({completion:.2f}%) "
                f"generated={batch_generated_count} appended={batch_written_count} cumulative_appended={accepted_pairs} "
                f"requests={state['requests_in_window']}/{state['rpm_limit']} "
                f"tokens={state['tokens_in_window']}/{state['tpm_limit']}"
            )
    finally:
        output_handle.close()

    stats = {
        "input_records": len(source_rows),
        "generated_candidates": total_generated,
        "accepted_pairs": accepted_pairs,
        "failed_generations": failures,
        "acceptance_rate": (accepted_pairs / total_generated) if total_generated else 0.0,
        "model_name": config.model_name,
        "prompt_version": config.prompt_version,
        "candidates_per_record": config.candidates_per_record,
        "target_entries_per_request": config.target_entries_per_request,
        "rpm_limit": config.rpm_limit,
        "tpm_limit": config.tpm_limit,
        "rpd_limit": config.rpd_limit,
        "requested_batches": requested_batches,
        "rate_limit_wait_seconds": round(rate_limit_wait_seconds, 3),
        "rate_limit_resets": rate_limit_resets,
        "rolling_window_seconds": config.rolling_window_seconds,
        "max_retries": config.max_retries,
        "backoff_base_seconds": config.backoff_base_seconds,
        "input_path": str(config.input_path),
        "output_path": str(config.output_path),
    }

    config.stats_path.parent.mkdir(parents=True, exist_ok=True)
    with config.stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    return stats


def _estimate_text_tokens(text: str) -> int:
    # Cheap approximation for rate limiting without tokenizer dependencies.
    return max(1, len(text) // 4)


def _estimate_request_tokens(rows: list[dict[str, Any]], candidates_per_record: int) -> int:
    base_prompt_tokens = 250
    output_tokens_per_candidate = 80
    total = base_prompt_tokens

    for row in rows:
        text = str(row.get("text", ""))
        total += _estimate_text_tokens(text)
        total += 25  # metadata overhead per row in prompt
        total += candidates_per_record * output_tokens_per_candidate

    return total
