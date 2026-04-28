#!/usr/bin/env python3
"""Compute precise anomaly detection metrics from saved AdaptCLIP predictions.

Reads prediction_manifest.jsonl + pixel_maps/*.npy + gt_masks/*.npy,
computes metrics with GPU torchmetrics (metrics/ module).
Pixel maps are streamed in batches to avoid OOM on large datasets.

Usage:
    # Basic: evaluate by class
    python compute_metrics.py \
        --manifest results/visa_10seed_1shot_predictions/prediction_manifest.jsonl \
        --output-dir results/visa_10seed_1shot_eval/

    # Custom split CSV
    python compute_metrics.py \
        --manifest .../prediction_manifest.jsonl \
        --output-dir .../eval/ \
        --split-csv my_split.csv --split-col img_path

    # Select groupings and metrics
    python compute_metrics.py \
        --manifest .../prediction_manifest.jsonl \
        --output-dir .../eval/ \
        --groupings cls_name specie_name \
        --metrics I-AUROC P-AUROC P-AUPRO
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .metrics import AUROC, AUPR, AUPRO, F1Max

SUPPORTED_METRICS = [
    'I-AUROC', 'I-AP', 'I-F1max',
    'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO',
]


@dataclass(slots=True)
class Record:
    index: int
    sample_id: str
    cls_name: str
    specie_name: str
    anomaly: int
    img_path: str
    pred_score: float
    pixel_map_path: str
    gt_mask_path: str
    pixel_map_shape: tuple[int, int]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            shape = d.get('pixel_map_shape', [0, 0])
            records.append(Record(
                index=int(d['index']),
                sample_id=str(d.get('sample_id', '')),
                cls_name=str(d['cls_name']),
                specie_name=str(d.get('specie_name', '')),
                anomaly=int(d['anomaly']),
                img_path=str(d['img_path']),
                pred_score=float(d['pred_score']),
                pixel_map_path=str(d['pixel_map_path']),
                gt_mask_path=str(d.get('gt_mask_path', '')),
                pixel_map_shape=(int(shape[0]), int(shape[1])),
            ))
    if not records:
        raise ValueError(f'No records found in {path}')
    return records


def filter_by_split(records: list[Record], split_path: Path, img_path_col: str) -> list[Record]:
    """Keep only records whose img_path appears in the split CSV."""
    with split_path.open(encoding='utf-8') as f:
        allowed = {row[img_path_col] for row in csv.DictReader(f)}
    filtered = [r for r in records if r.img_path in allowed]
    if not filtered:
        raise ValueError(f'No records matched after filtering by {split_path}')
    print(f'  split filter: {len(records)} → {len(filtered)} samples')
    return filtered


def build_groups(records: list[Record], groupings: list[str]) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = {}

    if 'overall' in groupings:
        groups['overall'] = list(records)

    if 'cls_name' in groupings:
        buckets: dict[str, list[Record]] = defaultdict(list)
        for r in records:
            buckets[r.cls_name].append(r)
        groups.update(buckets)

    if 'specie_name' in groupings:
        # anomaly samples of that specie + normal samples from the same classes
        pos_by_specie: dict[str, list[Record]] = defaultdict(list)
        for r in records:
            if r.anomaly == 1:
                pos_by_specie[r.specie_name].append(r)
        for specie, pos in pos_by_specie.items():
            cls_set = {r.cls_name for r in pos}
            normal = [r for r in records if r.anomaly == 0 and r.cls_name in cls_set]
            groups[f'specie::{specie}'] = pos + normal

    if 'cls_name+specie_name' in groupings:
        pos_by_pair: dict[tuple[str, str], list[Record]] = defaultdict(list)
        for r in records:
            if r.anomaly == 1:
                pos_by_pair[(r.cls_name, r.specie_name)].append(r)
        for (cls_name, specie), pos in pos_by_pair.items():
            normal = [r for r in records if r.anomaly == 0 and r.cls_name == cls_name]
            groups[f'{cls_name}::{specie}'] = pos + normal

    return groups


def load_batch(batch: list[Record], base_dir: Path, device: torch.device):
    """Load a batch of pixel maps and gt masks as tensors."""
    pixel_maps = np.stack([
        np.load(base_dir / r.pixel_map_path).astype(np.float32) for r in batch
    ])
    gt_masks = np.stack([
        np.load(base_dir / r.gt_mask_path).astype(np.uint8) for r in batch
    ])
    return (
        torch.from_numpy(pixel_maps).to(device),   # [B, H, W] float32
        torch.from_numpy(gt_masks).to(device),     # [B, H, W] uint8
    )


# ---------------------------------------------------------------------------
# Metrics engine
# ---------------------------------------------------------------------------

class MetricsEngine:
    """Wraps all torchmetrics objects; supports reset/update/compute cycle."""

    def __init__(self, metrics: list[str], device: torch.device, aupro_num_thresholds: int | None = None) -> None:
        self.metrics = metrics
        aupro_kwargs = {'num_thresholds': aupro_num_thresholds} if aupro_num_thresholds else {}

        # image-level
        self.i_auroc = AUROC().to(device) if 'I-AUROC' in metrics else None
        self.i_aupr  = AUPR().to(device)  if 'I-AP'    in metrics else None
        self.i_f1max = F1Max().to(device) if 'I-F1max' in metrics else None

        # pixel-level
        self.p_auroc = AUROC().to(device)            if 'P-AUROC' in metrics else None
        self.p_aupr  = AUPR().to(device)             if 'P-AP'    in metrics else None
        self.p_f1max = F1Max().to(device)            if 'P-F1max' in metrics else None
        self.p_aupro = AUPRO(**aupro_kwargs).to(device) if 'P-AUPRO' in metrics else None

    def reset(self) -> None:
        for m in (self.i_auroc, self.i_aupr, self.i_f1max,
                  self.p_auroc, self.p_aupr, self.p_f1max, self.p_aupro):
            if m is not None:
                m.reset()

    def update_image(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        if self.i_auroc: self.i_auroc.update(scores, labels)
        if self.i_aupr:  self.i_aupr.update(scores, labels)
        if self.i_f1max: self.i_f1max.update(scores, labels)

    def update_pixel(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """preds/targets: [B, H, W]."""
        flat_p = preds.reshape(-1)
        flat_t = targets.reshape(-1)
        if self.p_auroc: self.p_auroc.update(flat_p, flat_t)
        if self.p_aupr:  self.p_aupr.update(flat_p, flat_t)
        if self.p_f1max: self.p_f1max.update(flat_p, flat_t)
        if self.p_aupro: self.p_aupro.update(preds, targets)   # AUPRO needs 2-D maps

    def compute(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.i_auroc: out['I-AUROC'] = self.i_auroc.compute().item()
        if self.i_aupr:  out['I-AP']    = self.i_aupr.compute().item()
        if self.i_f1max: out['I-F1max'] = self.i_f1max.compute().item()
        if self.p_auroc: out['P-AUROC'] = self.p_auroc.compute().item()
        if self.p_aupr:  out['P-AP']    = self.p_aupr.compute().item()
        if self.p_f1max: out['P-F1max'] = self.p_f1max.compute().item()
        if self.p_aupro: out['P-AUPRO'] = self.p_aupro.compute().item()
        return out


# ---------------------------------------------------------------------------
# Per-group evaluation
# ---------------------------------------------------------------------------

def evaluate_group(
    records: list[Record],
    base_dir: Path,
    engine: MetricsEngine,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    engine.reset()

    labels = torch.tensor([r.anomaly    for r in records], dtype=torch.long,    device=device)
    scores = torch.tensor([r.pred_score for r in records], dtype=torch.float32, device=device)

    summary: dict[str, object] = {
        'sample_count':  len(records),
        'anomaly_count': int(labels.sum().item()),
        'normal_count':  int((labels == 0).sum().item()),
    }

    # Need at least one sample from each class for all metrics
    if labels.unique().numel() < 2:
        for m in engine.metrics:
            summary[m] = float('nan')
        return summary

    engine.update_image(scores, labels)

    need_pixel = any(m.startswith('P-') for m in engine.metrics)
    if need_pixel:
        batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
        for batch in tqdm(batches, desc='  pixel maps', leave=False):
            preds, targets = load_batch(batch, base_dir, device)
            engine.update_pixel(preds, targets)

    summary.update(engine.compute())
    return summary


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict], metrics: list[str]) -> None:
    col_w = 30
    header = f"{'group':<{col_w}}" + ''.join(f'{m:>10}' for m in metrics)
    print('\n' + header)
    print('-' * len(header))
    for row in rows:
        group = str(row.get('group', ''))[:col_w]
        vals = ''.join(
            f"{row[m]*100:>9.1f}%" if isinstance(row.get(m), float) and row[m] == row[m]
            else f"{'N/A':>10}"
            for m in metrics
        )
        print(f'{group:<{col_w}}{vals}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute metrics from saved AdaptCLIP predictions')
    parser.add_argument('--manifest',   type=Path, required=True,
                        help='Path to prediction_manifest.jsonl')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Directory to write metrics.csv / metrics.json')
    parser.add_argument('--split-csv',  type=Path, default=None,
                        help='Optional CSV to filter samples (requires --split-col)')
    parser.add_argument('--split-col',  type=str, default='img_path',
                        help='Column in --split-csv that contains image paths (default: img_path)')
    parser.add_argument('--groupings',  type=str, nargs='+',
                        default=['overall', 'cls_name'],
                        choices=['overall', 'cls_name', 'specie_name', 'cls_name+specie_name'],
                        help='Grouping schemes (default: overall cls_name)')
    parser.add_argument('--metrics',    type=str, nargs='+',
                        default=SUPPORTED_METRICS, choices=SUPPORTED_METRICS,
                        help='Metrics to compute')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for streaming pixel maps (default: 32)')
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (default: cuda if available)')
    parser.add_argument('--aupro-num-thresholds', type=int, default=None,
                        help='Cap AUPRO threshold count to save VRAM on large datasets')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f'Loading manifest: {args.manifest}')
    records = load_manifest(args.manifest)
    print(f'  {len(records)} samples  |  device: {device}')

    if args.split_csv:
        records = filter_by_split(records, args.split_csv, args.split_col)

    groups = build_groups(records, args.groupings)
    print(f'  {len(groups)} groups from groupings={args.groupings}')

    engine = MetricsEngine(args.metrics, device, args.aupro_num_thresholds)
    base_dir = args.manifest.parent

    all_rows: list[dict] = []
    for group_name, group_records in tqdm(groups.items(), desc='Groups'):
        row: dict[str, object] = {'group': group_name}
        row.update(evaluate_group(group_records, base_dir, engine, args.batch_size, device))
        all_rows.append(row)
        write_csv(args.output_dir / 'metrics.csv', all_rows)  # incremental write

    json_path = args.output_dir / 'metrics.json'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    print_table(all_rows, args.metrics)
    print(f'\nSaved → {args.output_dir / "metrics.csv"}')
    print(f'        {json_path}')


if __name__ == '__main__':
    main()
