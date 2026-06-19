import gc
import importlib.util
import os
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_SCRIPT = Path(r"D:\yolo26\评定脚本最终版.py")
DEFAULT_WEIGHTS = BACKEND_DIR / "weights" / "best.pt"


def _load_legacy_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(
            f"找不到评片脚本: {script_path}. 请设置 RATING_SCRIPT_PATH 或把脚本放到该位置。"
        )

    spec = importlib.util.spec_from_file_location("legacy_rating_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载评片脚本: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _write_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"无法编码结果图片: {path}")
    encoded.tofile(str(path))


class RatingEngine:
    """Wrap the existing batch script as a reusable web inference engine."""

    def __init__(self, script_path: str | None = None, weights_path: str | None = None):
        self.script_path = Path(script_path or os.getenv("RATING_SCRIPT_PATH") or DEFAULT_LEGACY_SCRIPT)
        self.weights_path = Path(weights_path or os.getenv("MODEL_WEIGHTS_PATH") or DEFAULT_WEIGHTS)
        self._legacy = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()

    @property
    def class_names(self) -> list[str]:
        legacy = self._ensure_legacy()
        return list(getattr(legacy, "CLASS_NAMES", ["SL", "Pore", "Crack", "LP", "LF"]))

    def _ensure_legacy(self):
        if self._legacy is None:
            self._legacy = _load_legacy_module(self.script_path)
        return self._legacy

    def _resolve_weights(self) -> Path:
        if self.weights_path.exists():
            return self.weights_path

        legacy_default = self.script_path.parent / "models1" / "best.pt"
        if legacy_default.exists():
            return legacy_default

        raise FileNotFoundError(
            "找不到模型权重。请把 best.pt 放到 backend/weights/best.pt，"
            "或设置 MODEL_WEIGHTS_PATH 指向训练好的权重文件。"
        )

    def _ensure_model(self):
        if self._model is None:
            weights = self._resolve_weights()
            self._model = YOLO(str(weights))
        return self._model

    def predict_image(self, image_path: str | Path, output_dir: str | Path) -> dict:
        with self._lock:
            return self._predict_image_locked(Path(image_path), Path(output_dir))

    def _predict_image_locked(self, image_path: Path, output_dir: Path) -> dict:
        legacy = self._ensure_legacy()
        model = self._ensure_model()

        frame_orig = _read_image(image_path)
        if frame_orig is None:
            raise ValueError(f"无法读取图片: {image_path.name}")

        original_height, original_width = frame_orig.shape[:2]
        was_rotated = False

        if getattr(legacy, "ROTATE_ADAPTATION_ENABLED", True) and original_height > original_width * 1.1:
            frame_proc = cv2.rotate(frame_orig, cv2.ROTATE_90_CLOCKWISE)
            was_rotated = True
        else:
            frame_proc = frame_orig

        proc_height, proc_width = frame_proc.shape[:2]
        device = self._device

        if getattr(legacy, "INTELLIGENT_ENHANCEMENT_ENABLED", True):
            base_image = legacy.find_optimal_enhancement(
                model,
                frame_proc,
                getattr(legacy, "INTELLIGENT_ENHANCE_ALPHAS", [1.0]),
                device,
            )
        else:
            base_image = frame_proc.copy()

        inference_tasks = [{"name": "full_image", "image": base_image, "offset_xy": (0, 0)}]
        if getattr(legacy, "ZOOM_AND_SPLIT_ENABLED", True):
            split_count = getattr(legacy, "ZOOM_SPLIT_COUNT", 5)
            overlap_ratio = getattr(legacy, "ZOOM_SPLIT_OVERLAP_RATIO", 0.2)
            num_overlap = max(0, split_count - 1)
            base_w = proc_width / (split_count - overlap_ratio * num_overlap) if num_overlap > 0 else float(proc_width)
            overlap_px = int(base_w * overlap_ratio)
            x_start, split_index = 0, 0
            while x_start < proc_width:
                x_end = min(int(x_start + base_w), proc_width)
                if x_end > x_start:
                    inference_tasks.append(
                        {
                            "name": f"split_{split_index + 1}",
                            "image": base_image[:, x_start:x_end],
                            "offset_xy": (x_start, 0),
                        }
                    )
                    split_index += 1
                if x_end >= proc_width:
                    break
                x_start = x_end - overlap_px

        all_raw_detections = []
        for task in inference_tasks:
            task_img = task["image"]
            offset = task["offset_xy"]
            task_width = task_img.shape[1]
            images_to_process = [{"img": task_img, "flipped": False}]

            if getattr(legacy, "TEST_TIME_AUGMENTATION_ENABLED", True):
                tta_config = getattr(legacy, "TTA_CONFIG", {})
                if tta_config.get("horizontal_flip"):
                    images_to_process.append({"img": cv2.flip(task_img, 1), "flipped": True})
                for factor in tta_config.get("brightness_adjust", []):
                    images_to_process.append(
                        {"img": cv2.convertScaleAbs(task_img, alpha=factor, beta=0), "flipped": False}
                    )
                for gamma in tta_config.get("gamma_adjust", []):
                    images_to_process.append({"img": legacy.apply_gamma_correction(task_img, gamma), "flipped": False})

            for item in images_to_process:
                results = model.predict(source=item["img"], conf=0.01, device=device, verbose=False)
                if results and results[0].boxes:
                    for index in range(len(results[0].boxes)):
                        box = results[0].boxes.xyxy[index].cpu().numpy()
                        conf = results[0].boxes.conf[index].cpu().item()
                        cls_id = int(results[0].boxes.cls[index].cpu().item())
                        deaug_box = legacy.deaugment_box(box, task_width, item["flipped"])
                        all_raw_detections.append(
                            {
                                "box": [
                                    float(deaug_box[0] + offset[0]),
                                    float(deaug_box[1] + offset[1]),
                                    float(deaug_box[2] + offset[0]),
                                    float(deaug_box[3] + offset[1]),
                                ],
                                "conf": float(conf),
                                "cls_id": cls_id,
                            }
                        )

        class_names = self.class_names
        thresholds = getattr(legacy, "PER_CLASS_THRESHOLDS", {"default": 0.1})
        detections = [
            det
            for det in all_raw_detections
            if det["cls_id"] < len(class_names)
            and det["conf"] >= thresholds.get(class_names[det["cls_id"]], thresholds.get("default", 1.0))
        ]

        if getattr(legacy, "CROSS_CLASS_SUPPRESSION_ENABLED", True) and len(detections) > 1:
            detections = self._apply_cross_class_suppression(legacy, detections, class_names)

        if detections:
            boxes = np.array([det["box"] for det in detections])
            scores = np.array([det["conf"] for det in detections])
            classes = np.array([det["cls_id"] for det in detections])
            final_indices = legacy.non_max_suppression(
                boxes,
                scores,
                classes,
                getattr(legacy, "FINAL_NMS_IOU_THRESHOLD", 0.45),
            )
            detections = [detections[index] for index in final_indices]

        if getattr(legacy, "POSITIONAL_AWARE_SUPPRESSION_ENABLED", True):
            detections = legacy.apply_positional_suppression(
                detections,
                proc_width,
                getattr(legacy, "MIDDLE_ZONE_RATIO", 0.6),
                class_names,
            )
        if getattr(legacy, "CONTAINMENT_RECLASSIFICATION_ENABLED", True):
            detections = legacy.apply_containment_reclassification(detections, class_names)
        if getattr(legacy, "SHAPE_RECLASSIFICATION_ENABLED", True):
            detections = legacy.apply_shape_based_reclassification(
                detections,
                class_names,
                getattr(legacy, "PORE_LIKE_ASPECT_RATIO_MIN", 0.70),
                getattr(legacy, "PORE_LIKE_ASPECT_RATIO_MAX", 1.40),
            )
        if getattr(legacy, "CONFIDENCE_RECLASSIFICATION_ENABLED", True):
            detections = legacy.apply_confidence_based_reclassification(detections, class_names)

        if getattr(legacy, "POST_MERGE_ENABLED", True):
            detections_after_v46 = legacy.merge_nearby_boxes(
                detections,
                getattr(legacy, "MERGE_MAX_DISTANCE_X", 30),
                getattr(legacy, "MERGE_MAX_DISTANCE_Y", 15),
            )
        else:
            detections_after_v46 = detections

        if getattr(legacy, "EXPERT_REVIEW_ENABLED", True):
            gray_proc = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY) if len(frame_proc.shape) > 2 else frame_proc
            grad_x = cv2.Sobel(gray_proc, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_proc, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            detections_after_review = legacy.expert_review_module(
                detections_after_v46,
                frame_proc,
                grad_mag,
                class_names,
            )
        else:
            detections_after_review = detections_after_v46

        final_results = []
        if was_rotated:
            for det in detections_after_review:
                det_copy = det.copy()
                det_copy["box"] = legacy.de_rotate_box(det["box"], original_height)
                final_results.append(det_copy)
        else:
            final_results = detections_after_review

        final_results.sort(key=lambda det: det["box"][0])
        annotated_image = self._draw_results(legacy, frame_orig, final_results, class_names)
        output_path = output_dir / f"{image_path.stem}_annotated.jpg"
        _write_image(output_path, annotated_image)

        detections_payload = []
        for det in final_results:
            cls_id = int(det["cls_id"])
            x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
            detections_payload.append(
                {
                    "id": str(uuid.uuid4()),
                    "class_id": cls_id,
                    "class_name": class_names[cls_id] if cls_id < len(class_names) else str(cls_id),
                    "confidence": round(float(det["conf"]), 4),
                    "bbox": [x1, y1, x2, y2],
                    "status": "unconfirmed",
                    "remark": "",
                    "deleted": False,
                }
            )

        gc.collect()
        return {
            "annotated_path": str(output_path),
            "detections": detections_payload,
            "width": int(original_width),
            "height": int(original_height),
            "was_rotated": was_rotated,
        }

    def _apply_cross_class_suppression(self, legacy, detections: list[dict], class_names: list[str]) -> list[dict]:
        to_suppress = set()
        detections = sorted(detections, key=lambda det: det["conf"], reverse=True)
        for i, det_a in enumerate(detections):
            if i in to_suppress:
                continue
            for j in range(i + 1, len(detections)):
                if j in to_suppress:
                    continue
                det_b = detections[j]
                class_a = class_names[det_a["cls_id"]]
                class_b = class_names[det_b["cls_id"]]
                for rule_a, rule_b, iou_threshold in getattr(legacy, "SUPPRESSION_RULES", []):
                    if {class_a, class_b} == {rule_a, rule_b}:
                        if self._iou(det_a["box"], det_b["box"]) > iou_threshold:
                            to_suppress.add(j)
        return [det for index, det in enumerate(detections) if index not in to_suppress]

    @staticmethod
    def _iou(box_a, box_b) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return float(inter / (area_a + area_b - inter + 1e-8))

    def _draw_results(self, legacy, image, detections: list[dict], class_names: list[str]):
        vis_frame = image.copy()
        height, width = vis_frame.shape[:2]
        class_colors = getattr(
            legacy,
            "CLASS_COLORS",
            [(0, 165, 255), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255)],
        )
        label_positions = {}

        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
            cls_id = int(det["cls_id"])
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            color = class_colors[cls_id % len(class_colors)]
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, getattr(legacy, "BOX_THICKNESS", 2))

            label_text = f"{name} {det['conf']:.2f}"
            font_scale = getattr(legacy, "LABEL_FONT_SCALE", 0.7)
            font_thickness = getattr(legacy, "LABEL_FONT_THICKNESS", 2)
            label_padding = getattr(legacy, "LABEL_PADDING", 10)
            (text_w, text_h), _ = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )
            bg_w = text_w + 2 * label_padding
            bg_h = text_h + 2 * label_padding
            lx = x1 + (x2 - x1 - bg_w) // 2
            ly = y1 - 20 - bg_h
            lx = max(5, min(lx, width - bg_w - 5))

            final_y, collided, attempts = ly, True, 0
            while collided and attempts < 20:
                collided = False
                for px, (py, pw, ph) in label_positions.items():
                    overlap_x = not (lx + bg_w < px or lx > px + pw)
                    overlap_y = not (final_y + bg_h < py or final_y > py + ph)
                    if overlap_x and overlap_y:
                        collided = True
                        final_y -= getattr(legacy, "VERTICAL_SHIFT_ON_OVERLAP", 30)
                        break
                attempts += 1
            if final_y < 0:
                final_y = y2 + 20
            ly = final_y
            label_positions[lx] = (ly, bg_w, bg_h)

            cv2.rectangle(vis_frame, (lx, ly), (lx + bg_w, ly + bg_h), getattr(legacy, "LABEL_BACKGROUND_COLOR", (255, 255, 255)), -1)
            cv2.rectangle(vis_frame, (lx, ly), (lx + bg_w, ly + bg_h), color, 2)
            cv2.putText(
                vis_frame,
                label_text,
                (lx + label_padding, ly + text_h + label_padding),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                getattr(legacy, "LABEL_TEXT_COLOR", (0, 0, 0)),
                font_thickness,
                cv2.LINE_AA,
            )
            cv2.arrowedLine(
                vis_frame,
                (lx + bg_w // 2, ly + bg_h if ly < y1 else ly),
                (int((x1 + x2) / 2), y1 if ly < y1 else y2),
                color,
                getattr(legacy, "ARROW_THICKNESS", 2),
                tipLength=0.3,
            )

        return vis_frame
