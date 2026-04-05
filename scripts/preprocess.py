from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback path when dependency is missing
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _fallback_load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _print_phase_quick_help() -> None:
    print("No pipeline phase selected.")
    print()
    print("Phases:")
    print("  --enable-unified-build    Build unified_records.jsonl from raw datasets.")
    print("  --enable-augmentation     Generate augmented_records.jsonl from unified records.")
    print("  --enable-emotion-filter   Filter augmented pairs and build complete_data.jsonl.")
    print()
    print("Examples:")
    print("  Unified build only:")
    print("    python scripts/preprocess.py --enable-unified-build")
    print("  Augmentation only:")
    print("    python scripts/preprocess.py --enable-augmentation")
    print("  Full pipeline:")
    print("    python scripts/preprocess.py --enable-unified-build --enable-augmentation --enable-emotion-filter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected data pipeline phases for EAFGLACL."
    )
    parser.add_argument(
        "--enable-unified-build",
        action="store_true",
        help="Build unified dataset artifacts from raw datasets.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".datasets_raw"),
        help="Root directory containing raw datasets (default: .datasets_raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Datasets"),
        help="Output directory for processed files (default: Datasets).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=None,
        help="Optional cap per dataset for fast smoke runs.",
    )
    parser.add_argument(
        "--enable-emotion",
        action="store_true",
        help="Annotate unified records with predicted emotion (unified-build phase only).",
    )
    parser.add_argument(
        "--enable-augmentation",
        action="store_true",
        help="Generate Gemini-based emotion-consistent augmentation pairs for unified dataset entries.",
    )
    parser.add_argument(
        "--enable-emotion-filter",
        action="store_true",
        help="Run emotion filtering and merge stage.",
    )
    parser.add_argument(
        "--augmentation-input",
        type=Path,
        default=None,
        help="Input JSONL for augmentation (default: <output-dir>/unified_records.jsonl).",
    )
    parser.add_argument(
        "--augmentation-output",
        type=Path,
        default=None,
        help="Output JSONL for accepted augmentation pairs (default: <output-dir>/augmented_records.jsonl).",
    )
    parser.add_argument(
        "--augmentation-stats-output",
        type=Path,
        default=None,
        help="Output JSON for augmentation stats (default: <output-dir>/augmentation_stats.json).",
    )
    parser.add_argument(
        "--emotion-filter-input",
        type=Path,
        default=None,
        help="Input JSONL for emotion filtering (default: <output-dir>/augmented_records.jsonl).",
    )
    parser.add_argument(
        "--emotion-filter-output",
        type=Path,
        default=None,
        help="Output JSONL for filtered augmentations (default: <output-dir>/filtered_augmented_data.jsonl).",
    )
    parser.add_argument(
        "--emotion-complete-output",
        type=Path,
        default=None,
        help="Output JSONL for merged+shuffled complete data (default: <output-dir>/complete_data.jsonl).",
    )
    parser.add_argument(
        "--augmentation-max-records",
        type=int,
        default=None,
        help="Optional cap of source records for augmentation runs.",
    )
    parser.add_argument(
        "--augmentation-candidates-per-record",
        type=int,
        default=3,
        help="Number of Gemini candidates to generate per source record.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default=None,
        help="Gemini model name for augmentation generation (default: GEMINI_MODEL from .env).",
    )
    parser.add_argument(
        "--augmentation-temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for Gemini augmentation generation.",
    )
    parser.add_argument(
        "--augmentation-top-p",
        type=float,
        default=0.95,
        help="Top-p sampling for Gemini augmentation generation.",
    )
    parser.add_argument(
        "--augmentation-max-output-tokens",
        type=int,
        default=512,
        help="Maximum output tokens per Gemini generation.",
    )
    parser.add_argument(
        "--augmentation-target-entries",
        type=int,
        default=None,
        help="Target number of source entries to include in each Gemini request.",
    )
    parser.add_argument(
        "--augmentation-tpm-limit",
        type=int,
        default=None,
        help="Token-per-minute budget for augmentation requests.",
    )
    parser.add_argument(
        "--augmentation-rpm-limit",
        type=int,
        default=None,
        help="Requests-per-minute budget for augmentation requests.",
    )
    parser.add_argument(
        "--augmentation-rolling-window-seconds",
        type=int,
        default=None,
        help="Rolling window size in seconds for RPM/TPM tracking (default: 60).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load local secret/config values (e.g., GEMINI_API_KEY) from project .env.
    dotenv_path = PROJECT_ROOT / ".env"
    if load_dotenv is not None:
        load_dotenv(dotenv_path)
    else:
        _fallback_load_dotenv(dotenv_path)

    phase_flags = {
        "unified_build": args.enable_unified_build,
        "augmentation": args.enable_augmentation,
        "emotion_filter": args.enable_emotion_filter,
    }
    selected_phases = [phase for phase, enabled in phase_flags.items() if enabled]
    if not selected_phases:
        _print_phase_quick_help()
        raise SystemExit(1)

    result: dict[str, object] = {}
    result["phases"] = selected_phases

    if args.enable_unified_build:
        from data_prep.builder import build_unified_dataset
        from data_prep.inspection import write_inspection_report

        inspection = write_inspection_report(args.data_root, args.output_dir / "inspection.json")
        stats = build_unified_dataset(
            data_root=args.data_root,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_per_dataset=args.max_per_dataset,
            enable_emotion=args.enable_emotion,
            enable_fact=False,
        )

        result["inspection"] = inspection
        result["stats"] = stats

    if args.enable_augmentation:
        from augmentation.pipeline import AugmentationConfig, run_train_augmentation

        augmentation_input = args.augmentation_input or (args.output_dir / "unified_records.jsonl")
        augmentation_output = args.augmentation_output or (args.output_dir / "augmented_records.jsonl")
        augmentation_stats_output = args.augmentation_stats_output or (
            args.output_dir / "augmentation_stats.json"
        )

        env_target_entries = os.getenv("AUGMENTATION_TARGET_ENTRIES")
        env_tpm_limit = os.getenv("GEMINI_TPM_LIMIT") or os.getenv("TPM")
        env_rpm_limit = os.getenv("GEMINI_RPM_LIMIT") or os.getenv("RPM")
        env_rpd_limit = os.getenv("GEMINI_RPD_LIMIT") or os.getenv("RPD")
        env_window_seconds = os.getenv("GEMINI_ROLLING_WINDOW_SECONDS")

        target_entries = (
            args.augmentation_target_entries
            if args.augmentation_target_entries is not None
            else int(env_target_entries) if env_target_entries else 200
        )
        tpm_limit = (
            args.augmentation_tpm_limit
            if args.augmentation_tpm_limit is not None
            else int(env_tpm_limit) if env_tpm_limit else None
        )
        rpm_limit = (
            args.augmentation_rpm_limit
            if args.augmentation_rpm_limit is not None
            else int(env_rpm_limit) if env_rpm_limit else None
        )
        rpd_limit = int(env_rpd_limit) if env_rpd_limit else None
        rolling_window_seconds = (
            args.augmentation_rolling_window_seconds
            if args.augmentation_rolling_window_seconds is not None
            else int(env_window_seconds) if env_window_seconds else 60
        )
        model_name = args.gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        augmentation_stats = run_train_augmentation(
            AugmentationConfig(
                input_path=augmentation_input,
                output_path=augmentation_output,
                stats_path=augmentation_stats_output,
                max_records=args.augmentation_max_records,
                candidates_per_record=args.augmentation_candidates_per_record,
                model_name=model_name,
                temperature=args.augmentation_temperature,
                top_p=args.augmentation_top_p,
                max_output_tokens=args.augmentation_max_output_tokens,
                target_entries_per_request=target_entries,
                rpm_limit=rpm_limit,
                tpm_limit=tpm_limit,
                rpd_limit=rpd_limit,
                rolling_window_seconds=rolling_window_seconds,
            )
        )
        result["augmentation"] = augmentation_stats

    if args.enable_emotion_filter:
        from emotion.pipeline import run_emotion_filter_pipeline

        emotion_filter_input = args.emotion_filter_input or (args.output_dir / "augmented_records.jsonl")
        emotion_filter_output = args.emotion_filter_output or (
            args.output_dir / "filtered_augmented_data.jsonl"
        )
        emotion_complete_output = args.emotion_complete_output or (args.output_dir / "complete_data.jsonl")
        emotion_stats = run_emotion_filter_pipeline(
            augmented_input_path=emotion_filter_input,
            unified_input_path=args.output_dir / "unified_records.jsonl",
            filtered_output_path=emotion_filter_output,
            complete_output_path=emotion_complete_output,
            seed=args.seed,
        )
        result["emotion_filter"] = emotion_stats

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
