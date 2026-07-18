from __future__ import annotations

import os
from typing import Iterable

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_figure_triplet(fig, out_base: str) -> None:
    root, _ = os.path.splitext(out_base)
    fig.savefig(root + ".pdf", bbox_inches="tight")
    fig.savefig(root + ".svg", bbox_inches="tight")
    fig.savefig(root + ".png", dpi=600, bbox_inches="tight")


def save_csvs(csv_map: dict[str, pd.DataFrame], out_dir: str) -> None:
    ensure_dir(out_dir)
    for name, df in csv_map.items():
        df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


def save_alt_text(text: str, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")


def append_caption(path: str, title: str, caption: str) -> None:
    ensure_dir(os.path.dirname(path))
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, encoding="utf-8") as fh:
        fh.write(f"## {title}\n\n{caption.rstrip()}\n\n")


def reset_caption_file(path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("")
