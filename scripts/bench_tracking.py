"""Benchmark improved tracker on D:\\Dowland\\One work clip / source."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from app.core.tracking import AthleteTracker, save_tracking, tracking_quality, box_at_time
from app.ui.widgets.media_pool import is_video

MEDIA = Path(r"D:\Dowland\One")
OUT = MEDIA / "РаботаИИ" / "tracking_bench"


def pick_box(video: Path, t: float, color: str) -> tuple[float, float, float, float]:
    tracker = AthleteTracker(target_color=color)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("no frame")
    boxes = tracker.detect_people(frame)
    if not boxes:
        h, w = frame.shape[:2]
        return (w * 0.3, h * 0.15, w * 0.25, h * 0.7)
    # red often left / blue right depending on camera — score by belt
    from app.core.tracking import belt_color_score

    scored = sorted(
        boxes, key=lambda b: belt_color_score(frame, b[:4], color) + b[4] * 0.1, reverse=True
    )
    return scored[0][:4]


def write_overlay(video: Path, points, out_mp4: Path, max_sec: float = 45.0) -> None:
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if t > max_sec:
            break
        box = box_at_time(points, t)
        if box:
            x, y, bw, bh = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 140, 255), 2)
            cv2.putText(
                frame,
                f"t={t:.1f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 140, 255),
                2,
            )
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    work = MEDIA / "РаботаИИ" / "work_90s.mp4"
    if not work.exists():
        videos = [p for p in MEDIA.iterdir() if is_video(p)]
        work = videos[0]
    print("VIDEO", work)

    report = {
        "started": datetime.now().isoformat(),
        "video": str(work),
        "runs": [],
    }

    for color in ("red", "blue"):
        print(f"\n=== TRACK {color} ===")
        box = pick_box(work, 6.0, color)
        tracker = AthleteTracker(target_color=color)
        points = tracker.track_video(
            work,
            box,
            sample_fps=12.0,
            max_seconds=60.0,
            detect_every=2,
            target_color=color,
            progress=lambda p, m: print(f"  {p*100:5.1f}% {m}")
            if int(p * 20) != int((p - 0.01) * 20)
            else None,
        )
        q = tracking_quality(points)
        save_tracking(OUT / f"track_{color}.json", points)
        overlay = OUT / f"overlay_{color}.mp4"
        write_overlay(work, points, overlay, max_sec=45.0)
        run = {
            "color": color,
            "points": len(points),
            "quality": q,
            "overlay": str(overlay),
        }
        report["runs"].append(run)
        print("QUALITY", q)

    report["finished"] = datetime.now().isoformat()
    (OUT / "bench.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# Tracking benchmark",
        f"Video: `{work}`",
        "",
    ]
    for r in report["runs"]:
        q = r["quality"]
        md.append(
            f"- **{r['color']}**: points={r['points']}, coverage={q['coverage']*100:.1f}%, "
            f"gaps={q['gaps']*100:.1f}%, jitter={q['jitter']:.2f}"
        )
        md.append(f"  - overlay: `{r['overlay']}`")
    (OUT / "BENCH.md").write_text("\n".join(md), encoding="utf-8")
    print("\nWROTE", OUT)
    # Require solid coverage
    ok = all(r["quality"]["coverage"] >= 0.85 for r in report["runs"])
    print("PASS" if ok else "WEAK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
