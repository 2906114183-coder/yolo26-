<script setup>
import { computed, ref, watch, nextTick } from "vue";

const API_BASE = (() => {
  if (import.meta.env.PROD) return "";
  return import.meta.env.VITE_API_BASE || "";
})();

const classNames = ["SL", "Pore", "Crack", "LP", "LF"];
const statusOptions = ["unconfirmed", "confirmed", "false_positive"];
const statusLabels = { unconfirmed: "未确认", confirmed: "已确认", false_positive: "误检" };

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const MAX_FILE_COUNT = 50;

const selectedFiles = ref([]);
const job = ref(null);
const selectedImageId = ref("");
const isUploading = ref(false);
const isRefreshing = ref(false);
const message = ref("");

// Drag-and-drop state
const isDragOver = ref(false);

// Delete confirmation
const confirmDeleteId = ref(null);

// Image zoom/pan
const zoomScales = ref({});
const zoomPans = ref({});
const imgNaturalSizes = ref({});

// Image-detection linking
const hoveredDetectionId = ref(null);

// Debounce timers for remark edits
const remarkTimers = ref({});

// Detection-image overlay dimensions
const imgDisplaySizes = ref({});

const selectedImage = computed(() => {
  if (!job.value?.images?.length) return null;
  return job.value.images.find((image) => image.id === selectedImageId.value) || job.value.images[0];
});

const activeDetections = computed(() => {
  return (selectedImage.value?.detections || []).filter((d) => !d.deleted);
});

const hoveredDetection = computed(() => {
  if (!hoveredDetectionId.value) return null;
  return activeDetections.value.find((d) => d.id === hoveredDetectionId.value) || null;
});

const jobStats = computed(() => {
  const images = job.value?.images || [];
  const defectCount = images.reduce((sum, image) => sum + (image.active_detection_count || 0), 0);
  return { imageCount: images.length, defectCount };
});

function apiUrl(path) {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

function validateFiles(files) {
  if (files.length > MAX_FILE_COUNT) {
    message.value = `一次最多上传 ${MAX_FILE_COUNT} 张图片`;
    return false;
  }
  const oversized = files.filter((f) => f.size > MAX_FILE_SIZE);
  if (oversized.length) {
    message.value = `以下文件超过 20MB 限制: ${oversized.map((f) => f.name).join(", ")}`;
    return false;
  }
  return true;
}

function onFileChange(event) {
  const files = Array.from(event.target.files || []);
  if (!validateFiles(files)) return;
  selectedFiles.value = files;
}

function onDragEnter(e) {
  e.preventDefault();
  isDragOver.value = true;
}
function onDragLeave(e) {
  e.preventDefault();
  isDragOver.value = false;
}
function onDragOver(e) {
  e.preventDefault();
}
function onDrop(e) {
  e.preventDefault();
  isDragOver.value = false;
  const files = Array.from(e.dataTransfer.files || []);
  const imageFiles = files.filter((f) =>
    [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"].some((ext) => f.name.toLowerCase().endsWith(ext))
  );
  if (!imageFiles.length) {
    message.value = "请拖入图片文件（jpg, png, bmp, tif）";
    return;
  }
  if (imageFiles.length !== files.length) {
    message.value = `已过滤 ${files.length - imageFiles.length} 个非图片文件`;
  }
  if (!validateFiles(imageFiles)) return;
  selectedFiles.value = imageFiles;
}

async function uploadAndDetect() {
  if (!selectedFiles.value.length) {
    message.value = "请先选择至少一张射线底片。";
    return;
  }

  isUploading.value = true;
  message.value = "正在上传并调用模型评定...";
  const formData = new FormData();
  selectedFiles.value.forEach((file) => formData.append("files", file));

  try {
    const response = await fetch(`${API_BASE}/api/jobs`, {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "评定失败");
    job.value = payload;
    selectedImageId.value = payload.images?.[0]?.id || "";
    message.value = payload.error || "评定完成。";
  } catch (error) {
    message.value = error.message;
  } finally {
    isUploading.value = false;
    selectedFiles.value = [];
  }
}

async function refreshJob() {
  if (!job.value?.id) return;
  isRefreshing.value = true;
  try {
    const response = await fetch(`${API_BASE}/api/jobs/${job.value.id}`);
    job.value = await response.json();
    message.value = "已刷新";
  } catch (error) {
    message.value = error.message;
  } finally {
    isRefreshing.value = false;
  }
}

function localUpdateDetection(detectionId, patch) {
  if (!job.value?.images) return;
  for (const image of job.value.images) {
    for (const det of image.detections) {
      if (det.id === detectionId) {
        Object.assign(det, patch);
        image.active_detection_count = image.detections.filter((d) => !d.deleted).length;
        return;
      }
    }
  }
}

function localDeleteDetection(detectionId) {
  if (!job.value?.images) return;
  for (const image of job.value.images) {
    for (const det of image.detections) {
      if (det.id === detectionId) {
        det.deleted = true;
        det.status = "false_positive";
        image.active_detection_count = image.detections.filter((d) => !d.deleted).length;
        return;
      }
    }
  }
}

async function patchDetection(detection, patch) {
  // Optimistic local update
  localUpdateDetection(detection.id, patch);

  try {
    const response = await fetch(`${API_BASE}/api/jobs/${job.value.id}/detections/${detection.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    if (!response.ok) {
      const payload = await response.json();
      message.value = payload.detail || "修改失败";
      // Rollback: refresh full job
      await refreshJob();
      return;
    }
  } catch (error) {
    message.value = error.message;
    await refreshJob();
  }
}

function confirmDelete(detection) {
  confirmDeleteId.value = detection.id;
}

function cancelDelete() {
  confirmDeleteId.value = null;
}

async function executeDelete(detection) {
  confirmDeleteId.value = null;

  // Optimistic local update
  localDeleteDetection(detection.id);

  try {
    const response = await fetch(`${API_BASE}/api/jobs/${job.value.id}/detections/${detection.id}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      const payload = await response.json();
      message.value = payload.detail || "删除失败";
      await refreshJob();
      return;
    }
  } catch (error) {
    message.value = error.message;
    await refreshJob();
  }
}

function onRemarkInput(detection, event) {
  const value = event.target.value;
  if (remarkTimers.value[detection.id]) {
    clearTimeout(remarkTimers.value[detection.id]);
  }
  remarkTimers.value[detection.id] = setTimeout(() => {
    patchDetection(detection, { remark: value });
    delete remarkTimers.value[detection.id];
  }, 500);
}

function exportExcel() {
  if (!job.value?.id) return;
  window.open(`${API_BASE}/api/jobs/${job.value.id}/export.xlsx`, "_blank");
}

// Image zoom/pan
function getZoomKey(imageId, type) {
  return `${imageId}_${type}`;
}

function onImgWheel(e, imageId) {
  e.preventDefault();
  const key = imageId;
  const current = zoomScales.value[key] || 1;
  const delta = e.deltaY > 0 ? -0.2 : 0.2;
  const newScale = Math.max(0.5, Math.min(5, current + delta));
  zoomScales.value = { ...zoomScales.value, [key]: newScale };
}

function onImgMouseDown(e, imageId) {
  if ((zoomScales.value[imageId] || 1) <= 1) return;
  const startX = e.clientX;
  const startY = e.clientY;
  const key = `pan_${imageId}`;
  const current = zoomPans.value[key] || { x: 0, y: 0 };

  function onMove(ev) {
    const dx = ev.clientX - startX;
    const dy = ev.clientY - startY;
    zoomPans.value = {
      ...zoomPans.value,
      [key]: { x: current.x + dx, y: current.y + dy },
    };
  }
  function onUp() {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  }
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}

function onImgDblClick(imageId) {
  const key = imageId;
  const panKey = `pan_${imageId}`;
  zoomScales.value = { ...zoomScales.value, [key]: 1 };
  zoomPans.value = { ...zoomPans.value, [panKey]: { x: 0, y: 0 } };
}

function onImgLoad(e, imageId) {
  const img = e.target;
  imgNaturalSizes.value = {
    ...imgNaturalSizes.value,
    [imageId]: { w: img.naturalWidth, h: img.naturalHeight },
  };
  // Calculate display size
  nextTick(() => {
    const rect = img.getBoundingClientRect();
    imgDisplaySizes.value = {
      ...imgDisplaySizes.value,
      [imageId]: { w: rect.width, h: rect.height },
    };
  });
}

// For detection-image overlay
function calcOverlayStyle(detection) {
  const img = selectedImage.value;
  if (!img) return null;
  const nat = imgNaturalSizes.value[`annotated_${img.id}`];
  const disp = imgDisplaySizes.value[`annotated_${img.id}`];
  if (!nat || !disp) return null;
  const scaleX = disp.w / nat.w;
  const scaleY = disp.h / nat.h;
  const [x1, y1, x2, y2] = detection.bbox;
  return {
    left: `${x1 * scaleX}px`,
    top: `${y1 * scaleY}px`,
    width: `${(x2 - x1) * scaleX}px`,
    height: `${(y2 - y1) * scaleY}px`,
  };
}

function handleImgError(e) {
  e.target.style.display = "none";
  const fallback = e.target.parentElement.querySelector(".img-error-placeholder");
  if (fallback) fallback.style.display = "flex";
}
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <div>
        <h1>射线底片缺陷评片系统</h1>
        <p>ConDSeg + 自适应后处理 + YOLOv11s</p>
      </div>
      <div class="topbar-actions">
        <button class="ghost-button" type="button" :disabled="!job || isRefreshing" @click="refreshJob">
          {{ isRefreshing ? "刷新中..." : "刷新" }}
        </button>
        <button class="primary-button" type="button" :disabled="!job" @click="exportExcel">导出 Excel</button>
      </div>
    </header>

    <section
      class="upload-band"
      :class="{ 'drag-over': isDragOver }"
      @dragenter="onDragEnter"
      @dragover="onDragOver"
      @dragleave="onDragLeave"
      @drop="onDrop"
    >
      <label class="upload-box" :class="{ 'drag-active': isDragOver }">
        <span class="upload-title">
          <svg class="upload-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/>
            <line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          {{ isDragOver ? "松开以添加文件" : "选择或拖拽射线底片" }}
        </span>
        <span class="upload-meta">支持 jpg、png、bmp、tif，单张 ≤ 20MB，一次最多 50 张</span>
        <input type="file" multiple accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff" @change="onFileChange" />
      </label>
      <div class="upload-side">
        <div class="file-count">{{ selectedFiles.length }} 张待评定</div>
        <button class="primary-button" type="button" :disabled="isUploading" @click="uploadAndDetect">
          <svg v-if="isUploading" class="spin-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
          </svg>
          {{ isUploading ? "评定中..." : "开始评定" }}
        </button>
        <p class="message" :class="{ error: message?.includes('失败') || message?.includes('错误') }">{{ message }}</p>
      </div>
    </section>

    <section v-if="job" class="summary-strip">
      <div>
        <span>任务编号</span>
        <strong>{{ job.id.slice(0, 8) }}</strong>
      </div>
      <div>
        <span>底片数量</span>
        <strong>{{ jobStats.imageCount }}</strong>
      </div>
      <div>
        <span>缺陷数量</span>
        <strong>{{ jobStats.defectCount }}</strong>
      </div>
      <div>
        <span>任务状态</span>
        <strong :class="'status-' + job.status">{{ job.status }}</strong>
      </div>
    </section>

    <section v-if="job" class="workspace">
      <aside class="image-list">
        <button
          v-for="image in job.images"
          :key="image.id"
          type="button"
          :class="['image-row', { active: selectedImage?.id === image.id }]"
          @click="selectedImageId = image.id"
        >
          <span class="image-name">{{ image.original_name }}</span>
          <span :class="['status-pill', image.status]">{{ image.status }}</span>
          <span class="defect-count">{{ image.active_detection_count || 0 }} 项</span>
        </button>
      </aside>

      <section class="viewer">
        <template v-if="selectedImage">
          <div v-if="selectedImage.status === 'failed'" class="empty-state">
            {{ selectedImage.error || "该图片评定失败" }}
          </div>
          <template v-else>
            <figure
              :class="{ zoomable: true }"
              @wheel="(e) => onImgWheel(e, 'orig_' + selectedImage.id)"
              @mousedown="(e) => onImgMouseDown(e, 'orig_' + selectedImage.id)"
              @dblclick="onImgDblClick('orig_' + selectedImage.id)"
            >
              <figcaption>
                原图
                <span v-if="(zoomScales['orig_' + selectedImage.id] || 1) !== 1" class="zoom-badge">
                  {{ Math.round((zoomScales['orig_' + selectedImage.id] || 1) * 100) }}%
                </span>
              </figcaption>
              <div class="img-wrapper">
                <img
                  :src="apiUrl(selectedImage.original_url)"
                  :alt="selectedImage.original_name"
                  :style="{
                    transform: `scale(${zoomScales['orig_' + selectedImage.id] || 1}) translate(${zoomPans['pan_orig_' + selectedImage.id]?.x || 0}px, ${zoomPans['pan_orig_' + selectedImage.id]?.y || 0}px)`,
                    cursor: (zoomScales['orig_' + selectedImage.id] || 1) > 1 ? 'grab' : 'default',
                  }"
                  @load="(e) => onImgLoad(e, 'orig_' + selectedImage.id)"
                  @error="handleImgError"
                />
                <div class="img-error-placeholder" style="display:none">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                  </svg>
                  <span>图片加载失败</span>
                </div>
              </div>
            </figure>

            <figure
              :class="{ zoomable: true }"
              @wheel="(e) => onImgWheel(e, 'annotated_' + selectedImage.id)"
              @mousedown="(e) => onImgMouseDown(e, 'annotated_' + selectedImage.id)"
              @dblclick="onImgDblClick('annotated_' + selectedImage.id)"
            >
              <figcaption>
                模型标注图
                <span v-if="(zoomScales['annotated_' + selectedImage.id] || 1) !== 1" class="zoom-badge">
                  {{ Math.round((zoomScales['annotated_' + selectedImage.id] || 1) * 100) }}%
                </span>
              </figcaption>
              <div class="img-wrapper" style="position:relative">
                <img
                  :src="apiUrl(selectedImage.annotated_url)"
                  :alt="`${selectedImage.original_name} 标注结果`"
                  :style="{
                    transform: `scale(${zoomScales['annotated_' + selectedImage.id] || 1}) translate(${zoomPans['pan_annotated_' + selectedImage.id]?.x || 0}px, ${zoomPans['pan_annotated_' + selectedImage.id]?.y || 0}px)`,
                    cursor: (zoomScales['annotated_' + selectedImage.id] || 1) > 1 ? 'grab' : 'default',
                  }"
                  @load="(e) => onImgLoad(e, 'annotated_' + selectedImage.id)"
                  @error="handleImgError"
                />
                <div class="img-error-placeholder" style="display:none">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                  </svg>
                  <span>图片加载失败</span>
                </div>
                <!-- Detection overlay on annotated image -->
                <div
                  v-if="hoveredDetection"
                  class="detection-overlay"
                  :style="calcOverlayStyle(hoveredDetection)"
                ></div>
              </div>
            </figure>
          </template>
        </template>
      </section>

      <aside class="detections">
        <div class="panel-heading">
          <h2>缺陷明细</h2>
          <span>{{ activeDetections.length }} 项</span>
        </div>

        <div v-if="!activeDetections.length" class="empty-state">未检测到缺陷或已全部删除。</div>

        <div
          v-for="detection in activeDetections"
          :key="detection.id"
          class="detection-item"
          :class="{ highlighted: hoveredDetectionId === detection.id }"
          @mouseenter="hoveredDetectionId = detection.id"
          @mouseleave="hoveredDetectionId = null"
        >
          <div class="detection-main">
            <select
              :value="detection.class_name"
              @change="patchDetection(detection, { class_name: $event.target.value })"
            >
              <option v-for="name in classNames" :key="name" :value="name">{{ name }}</option>
            </select>
            <select :value="detection.status" @change="patchDetection(detection, { status: $event.target.value })">
              <option v-for="status in statusOptions" :key="status" :value="status">{{ statusLabels[status] || status }}</option>
            </select>
          </div>

          <dl>
            <div>
              <dt>置信度</dt>
              <dd>{{ detection.confidence }}</dd>
            </div>
            <div>
              <dt>坐标</dt>
              <dd>{{ detection.bbox.join(", ") }}</dd>
            </div>
          </dl>

          <textarea
            :value="detection.remark"
            rows="2"
            placeholder="备注"
            @input="(e) => onRemarkInput(detection, e)"
          ></textarea>

          <template v-if="confirmDeleteId === detection.id">
            <div class="confirm-delete-bar">
              <span>确认删除此缺陷？</span>
              <button class="danger-button small" type="button" @click="executeDelete(detection)">确认删除</button>
              <button class="ghost-button small" type="button" @click="cancelDelete">取消</button>
            </div>
          </template>
          <button v-else class="danger-button" type="button" @click="confirmDelete(detection)">删除</button>
        </div>
      </aside>
    </section>

    <section v-else class="placeholder">
      <h2>等待上传底片</h2>
      <p>选择图片后点击开始评定，系统会调用后端模型并在这里显示评片结果。</p>
    </section>
  </main>
</template>
