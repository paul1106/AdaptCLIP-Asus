"""Generate AdaptCLIP-compatible meta.json from split CSV files.

Usage:
    python dataset/make_meta_from_splits.py \\
        --dataset-root /project/.../Asus-Visa \\
        --split-dir    /project/.../split_ratio1to1_instanced_cross_topup \\
        --output       /project/.../Asus-Visa/meta_split.json

The three expected CSVs (train_normal.csv, test_good.csv, test_anomaly.csv)
must have columns: image, label, mask, component, instance, defect_type

Output meta.json format matches AdaptCLIP's existing convention:
    {
      "train": { "<cls>": [ {img_path, mask_path, cls_name, specie_name, anomaly}, ... ] },
      "test":  { "<cls>": [ ... ] }
    }
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def csv_to_records(csv_path: Path, split: str) -> dict[str, list[dict]]:
    """Read one CSV and return a dict keyed by component (cls_name)."""
    cls_records: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label       = row['label'].strip()
            anomaly     = 1 if label == 'anomaly' else 0
            mask_path   = row['mask'].strip() if anomaly else ''
            defect_type = row['defect_type'].strip()
            specie_name = '' if defect_type in ('', 'None') else defect_type
            cls_name    = row['component'].strip()

            cls_records[cls_name].append({
                'img_path':    row['image'].strip(),
                'mask_path':   mask_path,
                'cls_name':    cls_name,
                'specie_name': specie_name,
                'anomaly':     anomaly,
            })
    return dict(cls_records)


def merge_dicts(dicts: list[dict[str, list]]) -> dict[str, list]:
    merged: dict[str, list] = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            merged[k].extend(v)
    return dict(merged)


def main():
    parser = argparse.ArgumentParser(description='Build meta.json from split CSVs')
    parser.add_argument('--dataset-root', required=True,
                        help='Root directory of Asus-Visa dataset')
    parser.add_argument('--split-dir', required=True,
                        help='Directory containing train_normal.csv, test_good.csv, test_anomaly.csv')
    parser.add_argument('--output', default=None,
                        help='Output path for meta.json (default: <dataset-root>/meta_split.json)')
    args = parser.parse_args()

    split_dir    = Path(args.split_dir)
    dataset_root = Path(args.dataset_root)
    output_path  = Path(args.output) if args.output else dataset_root / 'meta_split.json'

    train_records = csv_to_records(split_dir / 'train_normal.csv', 'train')
    test_records  = merge_dicts([
        csv_to_records(split_dir / 'test_good.csv',    'test'),
        csv_to_records(split_dir / 'test_anomaly.csv', 'test'),
    ])

    # Sort test records: normal before anomaly within each class (cosmetic)
    for cls in test_records:
        test_records[cls].sort(key=lambda r: r['anomaly'])

    # Ensure train and test have the same class set
    all_cls = sorted(set(train_records) | set(test_records))
    meta = {
        'train': {c: train_records.get(c, []) for c in all_cls},
        'test':  {c: test_records.get(c, [])  for c in all_cls},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total_train = sum(len(v) for v in meta['train'].values())
    total_test  = sum(len(v) for v in meta['test'].values())
    n_anomaly   = sum(r['anomaly'] for v in meta['test'].values() for r in v)
    print(f'Wrote {output_path}')
    print(f'  classes : {len(all_cls)}')
    print(f'  train   : {total_train} samples')
    print(f'  test    : {total_test} samples ({n_anomaly} anomaly, {total_test - n_anomaly} normal)')


if __name__ == '__main__':
    main()
