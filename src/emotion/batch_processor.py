import json
from collections import defaultdict
from extractor import EmotionExtractor
from filter import filter_augmentations


class BatchProcessor:
    def __init__(self):
        print("⏳ Initializing Emotion Model...")
        self.extractor = EmotionExtractor()

    def _load_jsonl(self, filepath: str) -> list:
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def run(self, augmented_file: str, output_file: str):
        print(f"📂 Loading paired dataset from {augmented_file}...")
        aug_data = self._load_jsonl(augmented_file)

        grouped_augs = defaultdict(list)
        for aug in aug_data:
            source_id = aug.get("source_record_id")
            if source_id:
                grouped_augs[source_id].append(aug)

        print(f"⚙️ Processing {len(grouped_augs)} unique original texts...\n")

        with open(output_file, 'w', encoding='utf-8') as f_out:
            total_saved = 0

            for i, (source_id, aug_group) in enumerate(grouped_augs.items()):
                source_text = aug_group[0].get("source_text", "")

                # If there's no source text, skip this entire group
                if not source_text:
                    continue

                # 1. Extract the anchor emotion (This is the variable that went missing!)
                orig_dist = self.extractor.get_emotion_distribution(source_text)

                # 2. Extract all candidate emotions in one massive GPU batch
                candidate_texts = [rec.get("candidate_text", "") for rec in aug_group]
                candidate_dists = self.extractor.get_emotion_distributions_batch(candidate_texts)

                # 3. Filter using the math logic
                valid_augs = filter_augmentations(orig_dist, aug_group, candidate_dists)

                # 4. Clean up and save
                for valid_aug in valid_augs:
                    valid_aug.pop("source_text", None)
                    if "candidate_text" in valid_aug:
                        valid_aug["text"] = valid_aug.pop("candidate_text")

                    f_out.write(json.dumps(valid_aug, ensure_ascii=False) + '\n')
                    total_saved += 1

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(grouped_augs)} source records... Saved {total_saved} valid pairs.")

        print("\n" + "=" * 50)
        print(f"✅ FINISHED: Saved {total_saved} emotion-preserved pairs to {output_file}")
        print("=" * 50)