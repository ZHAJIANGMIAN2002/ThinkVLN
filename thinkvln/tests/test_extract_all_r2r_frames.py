from pathlib import Path

import cv2
import numpy as np

from scripts.extract_all_r2r_frames import extract_frames


def _write_video(video_path: Path, frames) -> None:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        6,
        (width, height),
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_extract_frames_skips_complete_episode(tmp_path: Path):
    video_root = tmp_path / "images"
    output_root = tmp_path / "r2r"
    rel_dir = Path("scene_r2r_000001")
    video_path = video_root / rel_dir / "trajectory.mp4"
    frame = np.zeros((4, 10, 3), dtype=np.uint8)
    _write_video(video_path, [frame])

    destination = output_root / rel_dir
    destination.mkdir(parents=True)
    (destination / "000000_rgb.jpg").write_bytes(b"rgb")
    (destination / "000000_map.jpg").write_bytes(b"map")

    result = extract_frames(video_path, video_root, output_root, overwrite=False, fixed_rgb_width=4)

    assert result["video"] == str(rel_dir)
    assert result["status"] == "skipped_complete"
    assert result["frames_total"] == 1
    assert result["frames_extracted"] == 0


def test_extract_frames_rebuilds_partial_episode_and_splits_frames(tmp_path: Path):
    video_root = tmp_path / "images"
    output_root = tmp_path / "r2r"
    rel_dir = Path("scene_r2r_000002")
    video_path = video_root / rel_dir / "trajectory.mp4"
    frames = []
    for idx in range(2):
        frame = np.zeros((6, 10, 3), dtype=np.uint8)
        frame[:, :4] = (idx + 1) * 20
        frame[:, 4:] = (idx + 1) * 40
        frames.append(frame)
    _write_video(video_path, frames)

    destination = output_root / rel_dir
    destination.mkdir(parents=True)
    (destination / "000000_rgb.jpg").write_bytes(b"old")

    result = extract_frames(video_path, video_root, output_root, overwrite=False, fixed_rgb_width=4)

    assert result["status"] == "rebuilt_partial"
    assert result["frames_total"] == 2
    assert result["frames_extracted"] == 4
    assert sorted(p.name for p in destination.iterdir()) == [
        "000000_map.jpg",
        "000000_rgb.jpg",
        "000001_map.jpg",
        "000001_rgb.jpg",
    ]

    rgb = cv2.imread(str(destination / "000001_rgb.jpg"))
    map_img = cv2.imread(str(destination / "000001_map.jpg"))
    assert rgb.shape[:2] == (6, 4)
    assert map_img.shape[:2] == (6, 6)


def test_extract_frames_rebuilds_when_existing_frame_count_is_short(tmp_path: Path):
    video_root = tmp_path / "images"
    output_root = tmp_path / "r2r"
    rel_dir = Path("scene_r2r_000003")
    video_path = video_root / rel_dir / "trajectory.mp4"
    frames = [np.full((6, 10, 3), 30 + idx, dtype=np.uint8) for idx in range(2)]
    _write_video(video_path, frames)

    destination = output_root / rel_dir
    destination.mkdir(parents=True)
    (destination / "000000_rgb.jpg").write_bytes(b"rgb")
    (destination / "000000_map.jpg").write_bytes(b"map")

    result = extract_frames(video_path, video_root, output_root, overwrite=False, fixed_rgb_width=4)

    assert result["status"] == "rebuilt_partial"
    assert result["frames_total"] == 2
    assert sorted(p.name for p in destination.iterdir()) == [
        "000000_map.jpg",
        "000000_rgb.jpg",
        "000001_map.jpg",
        "000001_rgb.jpg",
    ]
