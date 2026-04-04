import os
import json
import random
from batch_processor import BatchProcessor


def load_jsonl(filepath):
    """Safely loads a JSONL file into a list of dictionaries."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, filepath):
    """Saves a list of dictionaries to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    # 1. Resolve Dynamic Paths (Works no matter where you run the script from)
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../src/emotion
    base_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))  # .../EAFGLACL

    dataset_dir = os.path.join(base_dir, 'Dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    # Define exact file paths
    augmented_input_file = os.path.join(dataset_dir, 'augmented_records.jsonl')
    unified_input_file = os.path.join(dataset_dir, 'unified_records.jsonl')
    filtered_output_file = os.path.join(current_dir, 'filtered_augmented_data.jsonl')
    complete_output_file = os.path.join(dataset_dir, 'Complete_data.jsonl')

    # 2. Run the Emotion Filter
    print("\n" + "=" * 50)
    print("🚀 STEP 1: FILTERING AUGMENTATIONS")
    print("=" * 50)

    if not os.path.exists(augmented_input_file):
        print(f"❌ Error: Cannot find {augmented_input_file}")
        return

    processor = BatchProcessor()
    processor.run(augmented_input_file, filtered_output_file)

    # 3. Merge and Shuffle
    print("\n" + "=" * 50)
    print("🔀 STEP 2: MERGING AND SHUFFLING DATA")
    print("=" * 50)

    print(f"📂 Loading original data from unified_records.jsonl...")
    unified_data = load_jsonl(unified_input_file)

    print(f"📂 Loading filtered augmentations from filtered_augmented_data.jsonl...")
    filtered_data = load_jsonl(filtered_output_file)

    print(f"🔄 Merging {len(unified_data)} original + {len(filtered_data)} augmented records...")
    complete_data = unified_data + filtered_data

    print("🎲 Randomly shuffling the combined dataset...")
    random.seed(42)  # Optional: keeps the shuffle reproducible for research consistency
    random.shuffle(complete_data)

    print(f"💾 Saving {len(complete_data)} total records to Complete_data.jsonl...")
    save_jsonl(complete_data, complete_output_file)

    print("\n✅ DATA PREPARATION COMPLETE! Ready for Contrastive Learning.")


if __name__ == "__main__":
    main()