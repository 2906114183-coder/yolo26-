import json
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.excel_export import export_job_to_excel
from app.inference_adapter import RatingEngine


BACKEND_DIR = Path(__file__).resolve().parents[1]
RUNTIME_DIR = BACKEND_DIR / "runtime"
JOBS_DIR = RUNTIME_DIR / "jobs"
JOB_RETENTION_HOURS = int(os.getenv("JOB_RETENTION_HOURS", "24"))
STATIC_DIR = BACKEND_DIR / "static"
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_FILE_SIZE = 20 * 1024 * 1024

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=RUNTIME_DIR), name="files")

engine = RatingEngine()


class DetectionPatch(BaseModel):
    class_name: str | None = None
    status: str | None = None
    remark: str | None = None


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / "job.json"


def _save_job(job: dict) -> None:
    path = _job_path(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_job(job_id: str) -> dict:
    path = _job_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found or cleaned up")
    return json.loads(path.read_text(encoding="utf-8"))


def _public_url(path: str | Path) -> str:
    rel = Path(path).resolve().relative_to(RUNTIME_DIR.resolve())
    return "/files/" + rel.as_posix()


def _find_detection(job: dict, detection_id: str) -> tuple[dict, dict]:
    for image in job.get("images", []):
        for detection in image.get("detections", []):
            if detection.get("id") == detection_id:
                return image, detection
    raise HTTPException(status_code=404, detail="Defect record not found")


def _sanitize_error(exc: Exception) -> str:
    msg = str(exc)
    if not msg:
        return type(exc).__name__
    return f"{type(exc).__name__}: {msg.split(chr(10))[0][:200]}"


def _serialize_job(job: dict) -> dict:
    serialized = json.loads(json.dumps(job, ensure_ascii=False))
    for image in serialized.get("images", []):
        image["original_url"] = _public_url(image["original_path"])
        if image.get("annotated_path"):
            image["annotated_url"] = _public_url(image["annotated_path"])
        image["active_detection_count"] = len(
            [d for d in image.get("detections", []) if not d.get("deleted")]
        )
    return serialized


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "class_names": engine.class_names,
    }


def clean_up_old_jobs() -> int:
    now_ts = time.time()
    cutoff_ts = now_ts - JOB_RETENTION_HOURS * 3600
    cleaned = 0
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        job_file = job_dir / "job.json"
        if not job_file.exists():
            try:
                shutil.rmtree(job_dir)
                cleaned += 1
            except OSError:
                pass
            continue
        try:
            job_data = json.loads(job_file.read_text(encoding="utf-8"))
            updated_at = job_data.get("updated_at")
            if updated_at and datetime.fromisoformat(updated_at).timestamp() < cutoff_ts:
                shutil.rmtree(job_dir)
                cleaned += 1
        except (json.JSONDecodeError, ValueError, OSError):
            continue
    return cleaned


@app.post("/api/jobs")
async def create_job(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one image")

    for file in files:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file.filename} (max 20MB)",
            )
        await file.seek(0)

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    clean_up_old_jobs()
    upload_dir = job_dir / "uploads"
    result_dir = job_dir / "results"
    upload_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    job = {
        "id": job_id,
        "status": "running",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "images": [],
        "error": "",
    }
    _save_job(job)

    for file in files:
        original_name = Path(file.filename or "upload").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported image format: {original_name}")

        image_id = str(uuid.uuid4())
        image_path = upload_dir / f"{image_id}{suffix}"
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        image_record = {
            "id": image_id,
            "original_name": original_name,
            "original_path": str(image_path),
            "annotated_path": "",
            "status": "running",
            "error": "",
            "detections": [],
        }
        job["images"].append(image_record)
        _save_job(job)

        try:
            result = engine.predict_image(image_path, result_dir)
            image_record["annotated_path"] = result["annotated_path"]
            image_record["detections"] = result["detections"]
            image_record["width"] = result["width"]
            image_record["height"] = result["height"]
            image_record["was_rotated"] = result["was_rotated"]
            image_record["status"] = "done"
        except Exception as exc:
            image_record["status"] = "failed"
            image_record["error"] = _sanitize_error(exc)

        job["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _save_job(job)

    failed = [image for image in job["images"] if image["status"] == "failed"]
    job["status"] = "failed" if len(failed) == len(job["images"]) else "done"
    if failed:
        job["error"] = "Partial failure" if job["status"] == "done" else "All images failed"
    job["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_job(job)
    return _serialize_job(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _serialize_job(_load_job(job_id))


@app.patch("/api/jobs/{job_id}/detections/{detection_id}")
def update_detection(job_id: str, detection_id: str, patch: DetectionPatch):
    job = _load_job(job_id)
    _, detection = _find_detection(job, detection_id)
    if patch.class_name is not None:
        if patch.class_name not in engine.class_names:
            raise HTTPException(status_code=400, detail="Unknown defect class")
        detection["class_name"] = patch.class_name
        detection["class_id"] = engine.class_names.index(patch.class_name)
    if patch.status is not None:
        if patch.status not in {"unconfirmed", "confirmed", "false_positive"}:
            raise HTTPException(status_code=400, detail="Unknown status value")
        detection["status"] = patch.status
    if patch.remark is not None:
        detection["remark"] = patch.remark

    job["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_job(job)

    image_id = None
    for img in job.get("images", []):
        for det in img.get("detections", []):
            if det.get("id") == detection_id:
                image_id = img["id"]
                break

    return {
        "detection": detection,
        "job_id": job_id,
        "image_id": image_id,
    }


@app.delete("/api/jobs/{job_id}/detections/{detection_id}")
def delete_detection(job_id: str, detection_id: str):
    job = _load_job(job_id)
    _, detection = _find_detection(job, detection_id)
    detection["deleted"] = True
    detection["status"] = "false_positive"
    job["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _save_job(job)
    return {
        "deleted_id": detection_id,
        "job_id": job_id,
    }


@app.get("/api/jobs/{job_id}/export.xlsx")
def export_excel(job_id: str):
    job = _load_job(job_id)
    output_path = JOBS_DIR / job_id / "export.xlsx"
    export_job_to_excel(job, output_path)
    return FileResponse(
        output_path,
        filename=f"rating-job-{job_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="web")


@app.on_event("startup")
def _startup_cleanup():
    cleaned = clean_up_old_jobs()
    if cleaned:
        print(f"[cleanup] Removed {cleaned} expired job(s).")
