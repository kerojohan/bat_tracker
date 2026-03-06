import csv
import json
from pathlib import Path

import cv2
import numpy as np

import bat_tracker.pipeline as pipeline
from bat_tracker.video import VideoMeta


def test_pipeline_exports_files(monkeypatch, tmp_path: Path):
    frames = []
    for i in range(6):
        img = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(img, (10 + i * 5, 30), 3, 255, -1)
        frames.append((i, img))

    def fake_meta(_):
        return VideoMeta(path=Path("fake.mp4"), video_id="fake", fps=10.0, frame_count=6, width=64, height=64)

    def fake_bg(*args, **kwargs):
        return np.zeros((64, 64), dtype=np.uint8)

    def fake_iter(_):
        for item in frames:
            yield item

    monkeypatch.setattr(pipeline, "read_video_meta", fake_meta)
    monkeypatch.setattr(pipeline, "compute_background_median", fake_bg)
    monkeypatch.setattr(pipeline, "iter_gray_frames", fake_iter)

    out_dir = tmp_path / "out"
    meta = pipeline.run_pipeline("fake.mp4", str(out_dir), config_path=None)

    assert (out_dir / "background.png").exists()
    assert (out_dir / "tracks.csv").exists()
    assert (out_dir / "tracks_overlay.png").exists()
    assert (out_dir / "meta.json").exists()

    with (out_dir / "tracks.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == pipeline.CSV_COLUMNS
    assert len(rows) > 0

    with (out_dir / "meta.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["video"]["video_id"] == "fake"
    assert payload["metrics"]["frames_processed"] == 6
    assert meta["outputs"]["tracks_csv"].endswith("tracks.csv")
