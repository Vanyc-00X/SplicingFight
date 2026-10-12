from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.core.device import ultralytics_kwargs, device_label
from app.core.model_cache import get_yolo, reset_tracker_state


ProgressCallback = Callable[[float, str], None]


@dataclass
class TrackPoint:
    t: float
    x: float
    y: float
    w: float
    h: float
    conf: float = 1.0


def _clamp_box(
    box: tuple[float, float, float, float], fw: int, fh: int
) -> tuple[float, float, float, float]:
    x, y, w, h = box
    w = float(np.clip(w, 8, fw))
    h = float(np.clip(h, 8, fh))
    x = float(np.clip(x, 0, max(fw - w, 0)))
    y = float(np.clip(y, 0, max(fh - h, 0)))
    return x, y, w, h


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter + 1e-6
    return inter / union


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _crop_roi(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    fh, fw = frame.shape[:2]
    x, y, w, h = _clamp_box(box, fw, fh)
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def appearance_hist(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    roi = _crop_roi(frame, box)
    if roi is None or roi.size == 0:
        return None
    rh, rw = roi.shape[:2]
    y0, y1 = int(rh * 0.25), int(rh * 0.75)
    x0, x1 = int(rw * 0.15), int(rw * 0.85)
    torso = roi[y0:y1, x0:x1]
    if torso.size == 0:
        torso = roi
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.astype(np.float32)


def hist_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def belt_color_score(frame: np.ndarray, box: tuple[float, float, float, float], target: str | None) -> float:
    if not target:
        return 0.0
    roi = _crop_roi(frame, box)
    if roi is None or roi.size == 0:
        return 0.0
    rh, rw = roi.shape[:2]
    band = roi[int(rh * 0.45) : int(rh * 0.70), int(rw * 0.2) : int(rw * 0.8)]
    if band.size == 0:
        return 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    if target.lower() in {"red", "красный", "aka"}:
        m1 = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
        m2 = cv2.inRange(hsv, (165, 70, 50), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
    elif target.lower() in {"blue", "синий", "ao"}:
        mask = cv2.inRange(hsv, (90, 60, 40), (130, 255, 255))
    else:
        return 0.0
    return float(np.mean(mask > 0))


def make_template(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    roi = _crop_roi(frame, box)
    if roi is None or roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 128))


def template_score(frame: np.ndarray, box: tuple[float, float, float, float], tmpl: np.ndarray | None) -> float:
    if tmpl is None:
        return 0.0
    roi = _crop_roi(frame, box)
    if roi is None or roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (tmpl.shape[1], tmpl.shape[0]))
    # normalized correlation [-1,1] -> [0,1]
    a = gray.reshape(-1).astype(np.float32)
    b = tmpl.reshape(-1).astype(np.float32)
    if float(a.std()) < 1e-3 or float(b.std()) < 1e-3:
        return 0.0
    num = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(num):
        return 0.0
    return (num + 1.0) * 0.5


class KalmanBox:
    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 5e-2
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.ready = False

    def init(self, box: tuple[float, float, float, float]) -> None:
        cx, cy = _center(box)
        x, y, w, h = box
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        self.ready = True

    def predict(self) -> tuple[float, float, float, float]:
        pred = self.kf.predict().reshape(-1)
        cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
        return cx - w / 2, cy - h / 2, max(w, 8), max(h, 8)

    def correct(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        cx, cy = _center(box)
        meas = np.array([[cx], [cy], [box[2]], [box[3]]], dtype=np.float32)
        est = self.kf.correct(meas).reshape(-1)
        cx, cy, w, h = float(est[0]), float(est[1]), float(est[2]), float(est[3])
        return cx - w / 2, cy - h / 2, max(w, 8), max(h, 8)


class OpticalFlowBox:
    def __init__(self) -> None:
        self.prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = None

    def update(
        self, gray: np.ndarray, box: tuple[float, float, float, float]
    ) -> tuple[tuple[float, float, float, float], float]:
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return box, 0.0
        fh, fw = gray.shape[:2]
        x, y, w, h = _clamp_box(box, fw, fh)
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        if x2 - x1 < 12 or y2 - y1 < 12:
            self.prev_gray = gray
            return box, 0.0
        pad_x, pad_y = int(0.12 * (x2 - x1)), int(0.12 * (y2 - y1))
        rx1, ry1 = x1 + pad_x, y1 + pad_y
        rx2, ry2 = x2 - pad_x, y2 - pad_y
        if rx2 <= rx1 or ry2 <= ry1:
            self.prev_gray = gray
            return box, 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray[ry1:ry2, rx1:rx2],
            gray[ry1:ry2, rx1:rx2],
            None,
            0.5,
            2,
            12,
            2,
            5,
            1.1,
            0,
        )
        self.prev_gray = gray
        if flow is None or flow.size == 0:
            return box, 0.0
        dx = float(np.median(flow[..., 0]))
        dy = float(np.median(flow[..., 1]))
        mag = float(np.hypot(dx, dy))
        if mag > max(w, h) * 0.28:
            return box, 0.0
        nb = _clamp_box((x + dx, y + dy, w, h), fw, fh)
        conf = float(np.clip(1.0 - mag / (0.22 * max(w, h) + 1e-3), 0.15, 0.95))
        return nb, conf


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(box[2], 1.0) * max(box[3], 1.0)


def _search_roi(
    center_box: tuple[float, float, float, float],
    fw: int,
    fh: int,
    *,
    scale: float,
) -> tuple[int, int, int, int]:
    """Axis-aligned search window around the athlete (keeps judges out of YOLO)."""
    cx, cy = _center(center_box)
    w = max(center_box[2] * scale, fw * 0.12)
    h = max(center_box[3] * scale, fh * 0.18)
    x1 = int(np.clip(cx - w / 2, 0, fw - 1))
    y1 = int(np.clip(cy - h / 2, 0, fh - 1))
    x2 = int(np.clip(cx + w / 2, x1 + 8, fw))
    y2 = int(np.clip(cy + h / 2, y1 + 8, fh))
    return x1, y1, x2, y2


def _is_courtside(
    box: tuple[float, float, float, float],
    fw: int,
    fh: int,
    *,
    init_center: tuple[float, float],
    margin: float = 0.10,
) -> bool:
    """True for people glued to frame edges (corner judges) when target started on mat."""
    icx, icy = init_center
    # If user clicked near edge, don't apply courtside filter
    if icx < fw * 0.18 or icx > fw * 0.82 or icy < fh * 0.15 or icy > fh * 0.88:
        return False
    cx, cy = _center(box)
    near_x = cx < fw * margin or cx > fw * (1.0 - margin)
    near_y = cy < fh * margin or cy > fh * (1.0 - margin)
    return near_x or near_y


def _size_ok(
    box: tuple[float, float, float, float],
    ref: tuple[float, float, float, float],
    *,
    loose: bool,
) -> bool:
    ra, rb = _box_area(ref), _box_area(box)
    ratio = rb / max(ra, 1.0)
    lo, hi = (0.35, 2.8) if loose else (0.50, 2.0)
    if not (lo <= ratio <= hi):
        return False
    # Standing fighter is usually taller than wide; reject very squat boxes
    ref_ar = ref[3] / max(ref[2], 1.0)
    ar = box[3] / max(box[2], 1.0)
    if ref_ar >= 1.15 and ar < 0.85:
        return False
    return True


class AthleteTracker:
    """Local-ROI athlete tracker: YOLO only near the fighter — never latch onto corner judges."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        *,
        target_color: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.target_color = target_color

    def _model(self):
        return get_yolo(self.model_name)

    def _infer_kw(self) -> dict:
        from app.core.device import torch_device

        imgsz = 512 if torch_device() == "cuda" else 640
        return ultralytics_kwargs(imgsz=imgsz)

    def detect_people(
        self, frame: np.ndarray, *, conf: float = 0.25
    ) -> list[tuple[float, float, float, float, float]]:
        model = self._model()
        kw = self._infer_kw()
        results = model.predict(frame, classes=[0], conf=conf, iou=0.45, **kw)
        boxes: list[tuple[float, float, float, float, float]] = []
        if not results or results[0].boxes is None:
            return boxes
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            c = float(box.conf[0]) if box.conf is not None else 0.0
            x1, y1, x2, y2 = xyxy
            boxes.append((x1, y1, x2 - x1, y2 - y1, c))
        return boxes

    def detect_people_roi(
        self,
        frame: np.ndarray,
        roi: tuple[int, int, int, int],
        *,
        conf: float = 0.22,
    ) -> list[tuple[float, float, float, float, float]]:
        """Detect persons inside ROI; boxes returned in full-frame coordinates."""
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        local = self.detect_people(crop, conf=conf)
        out: list[tuple[float, float, float, float, float]] = []
        for x, y, w, h, c in local:
            out.append((x + x1, y + y1, w, h, c))
        return out

    def detect_people_tracked(
        self, frame: np.ndarray
    ) -> list[tuple[float, float, float, float, float, int | None]]:
        # Kept for click-bootstrap / tools; tracking loop uses ROI predict instead
        return [(*d, None) for d in self.detect_people(frame)]

    @staticmethod
    def pick_box(
        boxes: list[tuple[float, float, float, float, float]],
        click_xy: tuple[float, float],
    ) -> tuple[float, float, float, float, float] | None:
        cx, cy = click_xy
        best = None
        best_dist = 1e18
        for box in boxes:
            x, y, w, h, conf = box[:5]
            bx, by = x + w / 2, y + h / 2
            if x <= cx <= x + w and y <= cy <= y + h:
                dist = (bx - cx) ** 2 + (by - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best = (x, y, w, h, conf)
        if best is not None:
            return best
        for box in boxes:
            x, y, w, h, conf = box[:5]
            bx, by = x + w / 2, y + h / 2
            dist = (bx - cx) ** 2 + (by - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best = (x, y, w, h, conf)
        return best

    def _score_candidate(
        self,
        frame: np.ndarray,
        box: tuple[float, float, float, float],
        conf: float,
        current: tuple[float, float, float, float],
        predicted: tuple[float, float, float, float],
        ref_box: tuple[float, float, float, float],
        ref_hist: np.ndarray | None,
        tmpl: np.ndarray | None,
        init_center: tuple[float, float],
        fw: int,
        fh: int,
        *,
        loose: bool,
    ) -> float:
        if _is_courtside(box, fw, fh, init_center=init_center):
            return -1.0
        if not _size_ok(box, ref_box, loose=loose):
            return -1.0
        px, py = _center(predicted)
        cx, cy = _center(box)
        dist = float(np.hypot(cx - px, cy - py))
        diag = float(np.hypot(predicted[2], predicted[3]) + 1e-3)
        max_dist = (1.05 * diag) if loose else (0.55 * diag)
        # Absolute cap so we never teleport across the mat to a judge
        max_dist = min(max_dist, (0.22 if loose else 0.12) * max(fw, fh))
        if dist > max_dist:
            return -1.0
        app = (hist_similarity(ref_hist, appearance_hist(frame, box)) + 1.0) * 0.5
        tmpl_s = template_score(frame, box, tmpl)
        iou = max(_iou(current, box), _iou(predicted, box))
        belt = belt_color_score(frame, box, self.target_color)
        # Spatial continuity dominates — appearance of white gi is weak alone
        score = (
            0.28 * app
            + 0.22 * tmpl_s
            + 0.32 * iou
            + 0.10 * belt
            + 0.08 * conf
            + 0.15 * float(np.exp(-dist / (0.35 * diag)))
        )
        if app < 0.45 and iou < 0.15:
            score *= 0.25
        if self.target_color and belt < 0.015 and app < 0.62:
            score *= 0.4
        return score

    def _associate_local(
        self,
        frame: np.ndarray,
        detections: list[tuple[float, float, float, float, float]],
        current: tuple[float, float, float, float],
        predicted: tuple[float, float, float, float],
        ref_box: tuple[float, float, float, float],
        ref_hist: np.ndarray | None,
        tmpl: np.ndarray | None,
        init_center: tuple[float, float],
        *,
        loose: bool,
    ) -> tuple[tuple[float, float, float, float, float] | None, float]:
        fh, fw = frame.shape[:2]
        best = None
        best_score = -1.0
        min_score = 0.52 if loose else 0.46
        for det in detections:
            x, y, w, h, conf = det[:5]
            box = (x, y, w, h)
            score = self._score_candidate(
                frame,
                box,
                conf,
                current,
                predicted,
                ref_box,
                ref_hist,
                tmpl,
                init_center,
                fw,
                fh,
                loose=loose,
            )
            if score > best_score:
                best_score = score
                best = (x, y, w, h, conf)
        if best is None or best_score < min_score:
            return None, best_score
        return best, best_score

    def _bootstrap_box(
        self,
        frame: np.ndarray,
        initial_box: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        fh, fw = frame.shape[:2]
        current = _clamp_box(initial_box, fw, fh)
        dets0 = self.detect_people(frame, conf=0.20)
        best0 = None
        best_iou = 0.05
        icx, icy = _center(current)
        for det in dets0:
            box = det[:4]
            if _is_courtside(box, fw, fh, init_center=(icx, icy)):
                continue
            iou = _iou(current, box)
            cx, cy = _center(box)
            dist = float(np.hypot(cx - icx, cy - icy))
            if iou > best_iou and dist < 0.20 * max(fw, fh):
                best_iou = iou
                best0 = det
        if best0 is None:
            best_d = 1e18
            for det in dets0:
                box = det[:4]
                if _is_courtside(box, fw, fh, init_center=(icx, icy)):
                    continue
                cx, cy = _center(box)
                dist = float(np.hypot(cx - icx, cy - icy))
                if dist < best_d and dist < 0.25 * max(fw, fh):
                    best_d = dist
                    best0 = det
        if best0 is not None:
            current = _clamp_box(best0[:4], fw, fh)
        return current

    def _track_direction(
        self,
        cap: cv2.VideoCapture,
        *,
        start_frame: int,
        end_frame: int,
        step: int,
        initial_box: tuple[float, float, float, float],
        src_fps: float,
        record_step: int,
        detect_every: int,
        total_work: int,
        work_done: int,
        progress: ProgressCallback | None,
        label: str,
    ) -> tuple[list[TrackPoint], int]:
        """Track forward (step=+1) or backward (step=-1) between frame indices inclusive."""
        assert step in (-1, 1)
        points: list[TrackPoint] = []
        if (step > 0 and start_frame > end_frame) or (step < 0 and start_frame < end_frame):
            return points, work_done

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = cap.read()
        if not ok:
            return points, work_done

        fh, fw = frame.shape[:2]
        current = _clamp_box(initial_box, fw, fh)
        ref_box = current
        ref_hist = appearance_hist(frame, current)
        tmpl = make_template(frame, current)
        init_center = _center(current)
        last_good = current
        kalman = KalmanBox()
        kalman.init(current)
        flow = OpticalFlowBox()
        flow.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smooth = list(current)
        lost_frames = 0
        frame_idx = start_frame
        local_i = 0

        while True:
            if step > 0 and frame_idx > end_frame:
                break
            if step < 0 and frame_idx < end_frame:
                break

            fh, fw = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            predicted = kalman.predict() if kalman.ready else current
            of_box, of_conf = flow.update(gray, current)

            need_detect = local_i % detect_every == 0 or lost_frames >= 2 or of_conf < 0.25
            det_box = None
            det_conf = 0.0
            assoc_score = 0.0

            if need_detect:
                if lost_frames >= 20:
                    scale = 4.2
                elif lost_frames >= 8:
                    scale = 3.2
                else:
                    scale = 2.4
                roi = _search_roi(predicted if lost_frames < 5 else last_good, fw, fh, scale=scale)
                detections = self.detect_people_roi(frame, roi, conf=0.20)
                chosen, assoc_score = self._associate_local(
                    frame,
                    detections,
                    current,
                    predicted,
                    ref_box,
                    ref_hist,
                    tmpl,
                    init_center,
                    loose=lost_frames >= 8,
                )
                if chosen is not None:
                    det_box = chosen[:4]
                    det_conf = float(chosen[4])
                    lost_frames = 0
                    last_good = det_box
                    if assoc_score > 0.62:
                        nh = appearance_hist(frame, det_box)
                        if nh is not None and ref_hist is not None:
                            ref_hist = 0.94 * ref_hist + 0.06 * nh
                        elif nh is not None:
                            ref_hist = nh
                        nt = make_template(frame, det_box)
                        if nt is not None and tmpl is not None:
                            tmpl = cv2.addWeighted(tmpl, 0.92, nt, 0.08, 0)
                        elif nt is not None:
                            tmpl = nt
                        ref_box = tuple(
                            0.85 * ref_box[i] + 0.15 * det_box[i] for i in range(4)
                        )  # type: ignore[assignment]
                else:
                    lost_frames += 1

            max_step = 0.10 * max(fw, fh)
            if det_box is not None:
                jump = float(
                    np.hypot(
                        _center(det_box)[0] - _center(current)[0],
                        _center(det_box)[1] - _center(current)[1],
                    )
                )
                if jump > max_step:
                    if of_conf >= 0.28:
                        fused = kalman.correct(_clamp_box(of_box, fw, fh))
                        conf = of_conf * 0.65
                    else:
                        fused = _clamp_box(predicted, fw, fh)
                        conf = 0.18
                    lost_frames = max(lost_frames, 1)
                else:
                    if _iou(det_box, of_box) > 0.10 and of_conf >= 0.25:
                        fused = tuple(0.80 * det_box[i] + 0.20 * of_box[i] for i in range(4))
                    else:
                        fused = det_box
                    conf = max(det_conf, assoc_score, 0.40)
                    fused = kalman.correct(_clamp_box(fused, fw, fh))  # type: ignore[arg-type]
            elif of_conf >= 0.28 and lost_frames < 18:
                j = float(
                    np.hypot(
                        _center(of_box)[0] - _center(current)[0],
                        _center(of_box)[1] - _center(current)[1],
                    )
                )
                if j > max_step * 0.7:
                    fused = _clamp_box(predicted, fw, fh)
                    conf = 0.20
                else:
                    fused = kalman.correct(_clamp_box(of_box, fw, fh))
                    conf = of_conf * 0.8
            else:
                fused = _clamp_box(predicted, fw, fh)
                conf = 0.12

            if lost_frames >= 12:
                lgx, lgy = _center(last_good)
                fx, fy = _center(fused)
                if float(np.hypot(fx - lgx, fy - lgy)) > 0.18 * max(fw, fh):
                    fused = last_good
                    conf = 0.10

            alpha = 0.50 if conf > 0.6 else 0.32 if conf > 0.3 else 0.18
            smooth = [(1 - alpha) * smooth[i] + alpha * fused[i] for i in range(4)]
            current = _clamp_box((smooth[0], smooth[1], smooth[2], smooth[3]), fw, fh)

            if local_i % record_step == 0:
                t = frame_idx / src_fps
                points.append(
                    TrackPoint(
                        t=round(t, 4),
                        x=float(current[0]),
                        y=float(current[1]),
                        w=float(current[2]),
                        h=float(current[3]),
                        conf=float(np.clip(conf, 0.0, 1.0)),
                    )
                )

            work_done += 1
            if progress and total_work:
                progress(
                    min(0.02 + 0.97 * work_done / max(total_work, 1), 0.99),
                    f"{label} · кадр {frame_idx} · lost={lost_frames}",
                )

            next_idx = frame_idx + step
            if step > 0 and next_idx > end_frame:
                break
            if step < 0 and next_idx < end_frame:
                break
            # Seeking each frame is required for reliable reverse playback
            cap.set(cv2.CAP_PROP_POS_FRAMES, next_idx)
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx = next_idx
            local_i += 1

        return points, work_done

    def track_video(
        self,
        video_path: str | Path,
        initial_box: tuple[float, float, float, float],
        *,
        start_sec: float = 0.0,
        sample_fps: float = 10.0,
        max_seconds: float | None = None,
        detect_every: int = 3,
        target_color: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[TrackPoint]:
        """Track athlete starting at the click time (not blindly from frame 0)."""
        if target_color:
            self.target_color = target_color

        model = self._model()
        reset_tracker_state(model)
        if progress:
            progress(0.005, f"Модель готова · {device_label()}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            # fallback estimate
            frame_count = int((max_seconds or 120) * src_fps)
        duration = frame_count / src_fps
        end_sec = duration
        if max_seconds is not None:
            end_sec = min(end_sec, float(start_sec) + float(max_seconds))

        start_sec = float(np.clip(start_sec, 0.0, max(duration - 1e-3, 0.0)))
        start_frame = int(round(start_sec * src_fps))
        start_frame = int(np.clip(start_frame, 0, max(frame_count - 1, 0)))
        end_frame = int(min(frame_count - 1, max(0, int(end_sec * src_fps) - 1)))
        record_step = max(int(round(src_fps / max(sample_fps, 1.0))), 1)

        # Seed on the exact click frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, seed = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Не удалось прочитать кадр клика для трекинга")
        seed_box = self._bootstrap_box(seed, initial_box)
        if progress:
            progress(0.02, f"Старт с {start_sec:.1f}s · {device_label()}")

        # Backward only ~45s before click (full reverse on long fights is too slow)
        back_end_frame = max(0, start_frame - int(45.0 * src_fps))
        forward_frames = max(end_frame - start_frame + 1, 0)
        backward_frames = max(start_frame - back_end_frame, 0)
        total_work = max(forward_frames + backward_frames, 1)
        work_done = 0

        fwd, work_done = self._track_direction(
            cap,
            start_frame=start_frame,
            end_frame=end_frame,
            step=1,
            initial_box=seed_box,
            src_fps=src_fps,
            record_step=record_step,
            detect_every=detect_every,
            total_work=total_work,
            work_done=work_done,
            progress=progress,
            label="Вперёд",
        )
        back: list[TrackPoint] = []
        if start_frame > 0 and back_end_frame < start_frame:
            back, work_done = self._track_direction(
                cap,
                start_frame=start_frame - 1,
                end_frame=back_end_frame,
                step=-1,
                initial_box=seed_box,
                src_fps=src_fps,
                record_step=record_step,
                detect_every=detect_every,
                total_work=total_work,
                work_done=work_done,
                progress=progress,
                label="Назад",
            )

        cap.release()

        # Merge: backward points are reverse-time; sort by t and drop near-duplicates
        merged = sorted(back + fwd, key=lambda p: p.t)
        if not merged:
            x, y, w, h = seed_box
            merged = [TrackPoint(t=round(start_sec, 4), x=x, y=y, w=w, h=h, conf=1.0)]
        else:
            dedup: list[TrackPoint] = [merged[0]]
            for p in merged[1:]:
                if abs(p.t - dedup[-1].t) < 1e-4:
                    dedup[-1] = p
                else:
                    dedup.append(p)
            merged = dedup

        merged = self._smooth_points(merged)
        if progress:
            span = merged[-1].t - merged[0].t if len(merged) > 1 else 0.0
            progress(
                1.0,
                f"Трекинг готов · {len(merged)} точек · {span:.1f}s · {device_label()}",
            )
        return merged

    @staticmethod
    def _smooth_points(points: list[TrackPoint], win: int = 5) -> list[TrackPoint]:
        if len(points) < win:
            return points
        xs = np.array([p.x for p in points], dtype=np.float32)
        ys = np.array([p.y for p in points], dtype=np.float32)
        ws = np.array([p.w for p in points], dtype=np.float32)
        hs = np.array([p.h for p in points], dtype=np.float32)
        k = np.ones(win, dtype=np.float32) / win
        pad = win // 2

        def sm(arr: np.ndarray) -> np.ndarray:
            padded = np.pad(arr, (pad, pad), mode="edge")
            return np.convolve(padded, k, mode="valid")

        sxs, sys, sws, shs = sm(xs), sm(ys), sm(ws), sm(hs)
        return [
            TrackPoint(
                t=p.t,
                x=float(sxs[i]),
                y=float(sys[i]),
                w=float(sws[i]),
                h=float(shs[i]),
                conf=p.conf,
            )
            for i, p in enumerate(points)
        ]


def save_tracking(path: str | Path, points: list[TrackPoint]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in points]), encoding="utf-8")


def load_tracking(path: str | Path | None) -> list[TrackPoint]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    return [TrackPoint(**item) for item in json.loads(p.read_text(encoding="utf-8"))]


def scale_tracking(
    points: list[TrackPoint],
    *,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[TrackPoint]:
    if not points or not src_w or not src_h or not dst_w or not dst_h:
        return points
    if src_w == dst_w and src_h == dst_h:
        return points
    sx = dst_w / src_w
    sy = dst_h / src_h
    return [
        TrackPoint(t=p.t, x=p.x * sx, y=p.y * sy, w=p.w * sx, h=p.h * sy, conf=p.conf)
        for p in points
    ]


def box_at_time(points: list[TrackPoint], t: float) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    if t <= points[0].t:
        p = points[0]
        return p.x, p.y, p.w, p.h
    if t >= points[-1].t:
        p = points[-1]
        return p.x, p.y, p.w, p.h
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        if a.t <= t <= b.t:
            u = 0 if b.t == a.t else (t - a.t) / (b.t - a.t)
            u = u * u * (3 - 2 * u)
            return (
                a.x + (b.x - a.x) * u,
                a.y + (b.y - a.y) * u,
                a.w + (b.w - a.w) * u,
                a.h + (b.h - a.h) * u,
            )
    p = points[-1]
    return p.x, p.y, p.w, p.h


def tracking_quality(points: list[TrackPoint]) -> dict[str, float]:
    if not points:
        return {"coverage": 0.0, "jitter": 1e9, "gaps": 1.0}
    coverage = sum(1 for p in points if p.conf >= 0.3) / len(points)
    centers = np.array([_center((p.x, p.y, p.w, p.h)) for p in points], dtype=np.float32)
    if len(centers) > 2:
        vel = np.diff(centers, axis=0)
        accel = np.diff(vel, axis=0)
        jitter = float(np.mean(np.linalg.norm(accel, axis=1))) if len(accel) else 0.0
    else:
        jitter = 0.0
    low = sum(1 for p in points if p.conf < 0.25) / len(points)
    return {"coverage": float(coverage), "jitter": jitter, "gaps": float(low)}
