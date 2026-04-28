"""PredictionSaver: save per-sample inference outputs to disk.

Writes pixel anomaly maps + GT masks as individual .npy files,
and records metadata in a prediction_manifest.jsonl index file.
The saved format is consumed by compute_metrics.py.

Typical usage inside an inference loop::

    with PredictionSaver(pred_dir) as saver:
        for batch in dataloader:
            # ... run model ...
            saver.save_batch(
                pixel_maps   = pixel_anomaly_map,   # torch [B,H,W] or np [B,H,W]
                scores       = image_anomaly_score, # torch [B]     or np [B]
                gt_masks     = gt_mask,             # torch [B,H,W] or np [B,H,W]
                gt_labels    = gt_anomaly,          # torch [B]     or np [B]
                cls_names    = cls_name,            # list[str]
                img_paths    = img_path,            # list[str]
            )
    print(f'Saved {saver.count} samples → {saver.manifest_path}')
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _to_numpy(x) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy."""
    if hasattr(x, 'cpu'):          # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class PredictionSaver:
    """Save inference predictions to disk in a format compatible with compute_metrics.py.

    Args:
        pred_dir: Directory to write pixel_maps/, gt_masks/, and prediction_manifest.jsonl.
                  Created automatically if it does not exist.

    Example::

        saver = PredictionSaver('results/visa_1shot_predictions')
        for batch in dataloader:
            saver.save_batch(pixel_maps=..., scores=..., gt_masks=...,
                             gt_labels=..., cls_names=..., img_paths=...)
        saver.close()
    """

    def __init__(self, pred_dir: str | Path) -> None:
        self.pred_dir      = Path(pred_dir)
        self.pixel_map_dir = self.pred_dir / 'pixel_maps'
        self.gt_mask_dir   = self.pred_dir / 'gt_masks'
        self.manifest_path = self.pred_dir / 'prediction_manifest.jsonl'

        self.pixel_map_dir.mkdir(parents=True, exist_ok=True)
        self.gt_mask_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_file = self.manifest_path.open('w', encoding='utf-8')
        self._idx = 0

    # ------------------------------------------------------------------

    def save_batch(
        self,
        pixel_maps:  object,
        scores:      object,
        gt_masks:    object,
        gt_labels:   object,
        cls_names:   Sequence[str],
        img_paths:   Sequence[str],
        sample_ids:  Sequence[str] | None = None,
        specie_names: Sequence[str] | None = None,
    ) -> None:
        """Save one batch of predictions.

        Args:
            pixel_maps:   Pixel-level anomaly map, shape [B, H, W], float32.
            scores:       Image-level anomaly score, shape [B], float32.
            gt_masks:     Ground-truth binary mask, shape [B, H, W], uint8.
            gt_labels:    Ground-truth image label (0=normal, 1=anomaly), shape [B].
            cls_names:    Object class names, length B.
            img_paths:    Absolute image file paths, length B.
            sample_ids:   Optional sample identifiers; defaults to global index.
            specie_names: Optional defect type names; defaults to parent folder of img_path.
        """
        maps   = _to_numpy(pixel_maps).astype(np.float32)
        scores_ = _to_numpy(scores).astype(np.float64).ravel()
        masks  = _to_numpy(gt_masks).astype(np.uint8)
        labels = _to_numpy(gt_labels).astype(np.int32).ravel()
        B = len(cls_names)

        for b in range(B):
            fname = f'{self._idx:08d}.npy'
            np.save(self.pixel_map_dir / fname, maps[b])
            np.save(self.gt_mask_dir   / fname, masks[b])

            specie = (
                specie_names[b] if specie_names is not None
                else Path(img_paths[b]).parent.name
            )
            sid = str(sample_ids[b]) if sample_ids is not None else str(self._idx)

            record = {
                'index':           self._idx,
                'sample_id':       sid,
                'cls_name':        cls_names[b],
                'specie_name':     specie,
                'anomaly':         int(labels[b]),
                'img_path':        img_paths[b],
                'pred_score':      float(scores_[b]),
                'pixel_map_path':  f'pixel_maps/{fname}',
                'gt_mask_path':    f'gt_masks/{fname}',
                'pixel_map_shape': list(maps[b].shape),
            }
            self._manifest_file.write(json.dumps(record) + '\n')
            self._idx += 1

    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of samples saved so far."""
        return self._idx

    def close(self) -> Path:
        """Flush and close the manifest file. Returns the manifest path."""
        self._manifest_file.close()
        return self.manifest_path

    def __enter__(self) -> 'PredictionSaver':
        return self

    def __exit__(self, *_) -> None:
        self.close()
