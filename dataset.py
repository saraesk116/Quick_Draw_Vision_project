from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class QuickDrawClassFile:
    path: str
    label: int


class QuickDrawMemmapDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Quick, Draw! bitmap dataset backed by NumPy memmap.

    Each source file is a .npy with shape (N, 784) and typically dtype uint8.
    Samples are returned as float32 tensors shaped (1, 28, 28) normalized to [0, 1].
    Labels are returned as int64 scalar tensors.
    """

    def __init__(
        self,
        class_files: Sequence[QuickDrawClassFile | Tuple[str, int]],
        *,
        mmap_mode: str = "r",
        limit_per_class: int | None = None,
    ) -> None:
        super().__init__()

        normalized: List[QuickDrawClassFile] = []
        for item in class_files:
            if isinstance(item, QuickDrawClassFile):
                normalized.append(item)
            else:
                path, label = item
                normalized.append(QuickDrawClassFile(path=str(path), label=int(label)))

        if not normalized:
            raise ValueError("class_files must not be empty")

        self._files: List[QuickDrawClassFile] = normalized
        self._arrays: List[np.ndarray] = []
        self._labels: List[int] = []
        self._lengths: List[int] = []

        for f in self._files:
            array = np.load(str(Path(f.path)), mmap_mode=mmap_mode)
            if array.ndim != 2 or array.shape[1] != 784:
                raise ValueError(f"Unexpected array shape for {f.path}: {array.shape}")
            n = int(array.shape[0])
            if limit_per_class is not None:
                n = min(n, int(limit_per_class))
                array = array[:n]

            self._arrays.append(array)
            self._labels.append(int(f.label))
            self._lengths.append(n)

        self._cumulative: List[int] = []
        running = 0
        for n in self._lengths:
            running += n
            self._cumulative.append(running)

    def __len__(self) -> int:
        return self._cumulative[-1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        file_i = bisect_right(self._cumulative, idx)
        start = 0 if file_i == 0 else self._cumulative[file_i - 1]
        row_i = idx - start

        row = self._arrays[file_i][row_i]  # (784,)
        # Copy per-sample only (tiny) to float32 for PyTorch.
        x = torch.as_tensor(row, dtype=torch.float32).div_(255.0).view(1, 28, 28)
        y = torch.tensor(self._labels[file_i], dtype=torch.long)
        return x, y


def build_default_class_files(data_dir: str | Path = "data") -> List[QuickDrawClassFile]:
    data_dir = Path(data_dir)
    names = [
        ("full_numpy_bitmap_airplane.npy", 0),
        ("full_numpy_bitmap_banana.npy", 1),
        ("full_numpy_bitmap_cat.npy", 2),
        ("full_numpy_bitmap_alarm clock.npy", 3),
        ("full_numpy_bitmap_dolphin.npy", 4),
        ("full_numpy_bitmap_circle.npy", 5),
        ("full_numpy_bitmap_door.npy", 6),
        ("full_numpy_bitmap_eye.npy", 7),
        ("full_numpy_bitmap_moon.npy", 8),
        ("full_numpy_bitmap_donut.npy", 9),
    ]
    return [QuickDrawClassFile(path=str(data_dir / fname), label=label) for fname, label in names]
