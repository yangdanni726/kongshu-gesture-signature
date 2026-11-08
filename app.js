/* global requestAnimationFrame, cancelAnimationFrame, ResizeObserver */

const $ = (id) => document.getElementById(id);

const els = {
  cameraVideo: $("cameraVideo"),
  overlayCanvas: $("overlayCanvas"),
  cameraFrame: $("cameraFrame"),
  cameraPlaceholder: $("cameraPlaceholder"),
  placeholderTitle: $("placeholderTitle"),
  placeholderDetail: $("placeholderDetail"),
  cameraPill: $("cameraPill"),
  cameraPillText: $("cameraPillText"),
  liveTag: $("liveTag"),
  liveTagText: $("liveTagText"),
  directionTag: $("directionTag"),
  directionTagText: $("directionTagText"),
  handPill: $("handPill"),
  handPillText: $("handPillText"),
  frameHint: $("frameHint"),
  pinchBar: $("pinchBar"),
  pinchMark: $("pinchMark"),
  cameraStartBtn: $("cameraStartBtn"),
  cameraStartText: $("cameraStartText"),
  cameraStopBtn: $("cameraStopBtn"),
  mirrorToggle: $("mirrorToggle"),
  cameraPanel: document.querySelector(".camera-panel"),
  calibrateBtn: $("calibrateBtn"),
  calibrateBtnText: $("calibrateBtnText"),
  calibrationPanel: $("calibrationPanel"),
  calibrationProgress: $("calibrationProgress"),
  calibrationInstruction: $("calibrationInstruction"),
  calibrationHint: $("calibrationHint"),
  calibrationCaptureBtn: $("calibrationCaptureBtn"),
  calibrationSkipBtn: $("calibrationSkipBtn"),
  calibrationDefaultBtn: $("calibrationDefaultBtn"),
  calibrationState: $("calibrationState"),
  settingsCalibrateBtn: $("settingsCalibrateBtn"),
  resetCalibrationBtn: $("resetCalibrationBtn"),
  mapGain: $("mapGain"),
  mapGainValue: $("mapGainValue"),
  penWidth: $("penWidth"),
  penWidthValue: $("penWidthValue"),
  alertBanner: $("alertBanner"),
  alertText: $("alertText"),
  alertActionBtn: $("alertActionBtn"),
  cameraStateText: $("cameraStateText"),
  handStateText: $("handStateText"),
  writingStateText: $("writingStateText"),
  pinchStateText: $("pinchStateText"),
  writingPill: $("writingPill"),
  writingPillText: $("writingPillText"),
  directionText: $("directionText"),
  directionHint: $("directionHint"),
  inkCount: $("inkCount"),
  signaturePanel: document.querySelector(".signature-panel"),
  signatureShell: $("signatureShell"),
  signatureCanvas: $("signatureCanvas"),
  penCursor: $("penCursor"),
  signatureEmpty: $("signatureEmpty"),
  frozenBadge: $("frozenBadge"),
  writingTip: $("writingTip"),
  writingTipText: $("writingTipText"),
  toggleWritingBtn: $("toggleWritingBtn"),
  rewriteBtn: $("rewriteBtn"),
  undoBtn: $("undoBtn"),
  clearBtn: $("clearBtn"),
  exportOptions: $("exportOptions"),
  exportBtn: $("exportBtn"),
  exportBtnText: $("exportBtnText"),
  exportDialog: $("exportDialog"),
  exportPreviewStage: $("exportPreviewStage"),
  exportPreviewCanvas: $("exportPreviewCanvas"),
  previewBackgroundSwitch: $("previewBackgroundSwitch"),
  exportEdgeStatus: $("exportEdgeStatus"),
  downloadExportBtn: $("downloadExportBtn"),
  downloadExportBtnText: $("downloadExportBtnText"),
  pinchThreshold: $("pinchThreshold"),
  pinchThresholdValue: $("pinchThresholdValue"),
  releaseDelay: $("releaseDelay"),
  releaseDelayValue: $("releaseDelayValue"),
  smoothing: $("smoothing"),
  smoothingValue: $("smoothingValue"),
  directionMode: $("directionMode"),
  resetSettingsBtn: $("resetSettingsBtn"),
  helpButton: $("helpButton"),
  helpDialog: $("helpDialog"),
  toast: $("toast"),
  srStatus: $("srStatus"),
};

const exportRadios = Array.from(document.querySelectorAll('input[name="exportBackground"]'));
const previewBgButtons = Array.from(document.querySelectorAll(".preview-bg-option"));
const overlayCtx = els.overlayCanvas.getContext("2d");
const signatureCtx = els.signatureCanvas.getContext("2d");

const WASM_PATH = new URL("./assets/vendor/mediapipe/wasm", document.baseURI).href.replace(/\/$/, "");
const MODEL_PATH = new URL("./assets/models/hand_landmarker.task", document.baseURI).href;
const SETTINGS_KEY = "kongshu.signature.settings.v1";
const CALIBRATION_KEY = "kongshu.signature.calibration.v1";
const MIN_PINCH_RELEASE_GAP = 0.18;
const PINCH_RATIO_SMOOTHING = 0.38;
const MAX_METER_RATIO = 1.12;
const EXPORT_WIDTH = 2400;
const EXPORT_HEIGHT = 1000;
const SAFE_AREA = Object.freeze({ left: 0.06, top: 0.12, width: 0.88, height: 0.76 });
const TARGET_ASPECT = (SAFE_AREA.width * EXPORT_WIDTH) / (SAFE_AREA.height * EXPORT_HEIGHT);
const MAX_POINT_JUMP = EXPORT_WIDTH * 0.12;
const STABILITY_WAIT_MS = 260;
const CALIBRATION_STEPS = [
  { name: "左上角", shortName: "左上" },
  { name: "右上角", shortName: "右上" },
  { name: "左下角", shortName: "左下" },
  { name: "右下角", shortName: "右下" },
];

const DEFAULT_SETTINGS = Object.freeze({
  threshold: 0.38,
  releaseDelay: 140,
  smoothing: 55,
  mapGain: 1,
  penWidth: 3,
  directionMode: "follow",
  mirror: true,
});

const DEFAULT_ROI_X = Object.freeze({ minX: 0.09, maxX: 0.91 });

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const INDEX_CHAIN = new Set(["5-6", "6-7", "7-8"]);

let settings = { ...DEFAULT_SETTINGS };
let toastTimer = 0;

const state = {
  camera: "idle", // idle | requesting | live | error
  model: "idle", // idle | loading | ready | error
  writing: "idle", // idle | writing | finished
  stream: null,
  landmarker: null,
  vision: null,
  modelPromise: null,
  rafId: null,
  lastVideoTime: -1,
  handDetected: false,
  pinching: false,
  pinchRatio: null,
  pinchRatioSmoothed: null,
  pinchFrames: 0,
  releaseFrames: 0,
  releaseSince: null,
  blockUntilRelease: false,
  latestLandmarks: null,
  latestCursor: null,
  latestCanvasPoint: null,
  latestRawCanvasPoint: null,
  strokes: [],
  currentStroke: null,
  filters: null,
  overlayW: 0,
  overlayH: 0,
  overlayDpr: 1,
  signatureW: 0,
  signatureH: 0,
  alertSource: null,
  alertAction: null,
  exporting: false,
  calibrationRange: null,
  calibrationActive: false,
  calibrationStep: 0,
  calibrationPoints: [],
  drawBlockedUntil: 0,
  stabilityReason: null,
  lockedHandedness: null,
  lastWristPerceived: null,
  lastJumpNoticeAt: 0,
  exportBackground: "white",
};

/* -------------------------------------------------------------------------- */
/* Small utilities                                                            */
/* -------------------------------------------------------------------------- */

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function hasInk() {
  return state.strokes.length > 0 || Boolean(state.currentStroke && state.currentStroke.length);
}

function setText(element, text) {
  if (element.textContent !== text) element.textContent = text;
}

function setPill(pill, textEl, tone, text) {
  if (pill.dataset.state !== tone) pill.dataset.state = tone;
  setText(textEl, text);
}

function announce(message) {
  els.srStatus.textContent = "";
  window.setTimeout(() => {
    els.srStatus.textContent = message;
  }, 40);
}

function showToast(message, { error = false, duration = 2900 } = {}) {
  window.clearTimeout(toastTimer);
  els.toast.textContent = message;
  els.toast.classList.toggle("is-error", error);
  els.toast.classList.add("is-visible");
  toastTimer = window.setTimeout(() => {
    els.toast.classList.remove("is-visible");
  }, duration);
}

function showAlert(source, message, actionLabel, action) {
  state.alertSource = source;
  state.alertAction = action;
  els.alertText.textContent = message;
  els.alertActionBtn.textContent = actionLabel;
  els.alertActionBtn.hidden = !action;
  els.alertBanner.hidden = false;
}

function hideAlert(source) {
  if (source && state.alertSource !== source) return;
  state.alertSource = null;
  state.alertAction = null;
  els.alertBanner.hidden = true;
}

/* -------------------------------------------------------------------------- */
/* Settings                                                                   */
/* -------------------------------------------------------------------------- */

function normalizeSettings(raw) {
  const next = { ...DEFAULT_SETTINGS };
  if (!raw || typeof raw !== "object") return next;

  const threshold = Number(raw.threshold);
  if (Number.isFinite(threshold)) next.threshold = clamp(threshold, 0.2, 0.6);

  const releaseDelay = Number(raw.releaseDelay);
  if (Number.isFinite(releaseDelay)) {
    next.releaseDelay = clamp(Math.round(releaseDelay / 20) * 20, 60, 300);
  }

  const smoothing = Number(raw.smoothing);
  if (Number.isFinite(smoothing)) next.smoothing = clamp(Math.round(smoothing), 0, 100);

  const mapGain = Number(raw.mapGain);
  if (Number.isFinite(mapGain)) {
    next.mapGain = clamp(Math.round(mapGain * 20) / 20, 0.7, 1.3);
  }

  const penWidth = Number(raw.penWidth);
  if (Number.isFinite(penWidth)) next.penWidth = clamp(Math.round(penWidth), 1, 6);

  if (raw.directionMode === "follow" || raw.directionMode === "invert") {
    next.directionMode = raw.directionMode;
  }
  if (typeof raw.mirror === "boolean") next.mirror = raw.mirror;
  return next;
}

function loadSettings() {
  try {
    const raw = window.localStorage.getItem(SETTINGS_KEY);
    settings = normalizeSettings(raw ? JSON.parse(raw) : null);
  } catch (error) {
    console.warn("无法读取本地设置，将使用默认值。", error);
    settings = { ...DEFAULT_SETTINGS };
  }
}

function persistSettings() {
  try {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (error) {
    console.warn("无法保存本地设置。", error);
  }
}

function normalizeCalibration(raw) {
  if (!raw || typeof raw !== "object") return null;
  const minX = Number(raw.minX);
  const maxX = Number(raw.maxX);
  const minY = Number(raw.minY);
  const maxY = Number(raw.maxY);
  if (![minX, maxX, minY, maxY].every(Number.isFinite)) return null;
  if (minX < 0 || maxX > 1 || minY < 0 || maxY > 1) return null;
  if (maxX - minX < 0.12 || maxY - minY < 0.1) return null;
  return { minX, maxX, minY, maxY };
}

function loadCalibration() {
  try {
    const raw = window.localStorage.getItem(CALIBRATION_KEY);
    state.calibrationRange = normalizeCalibration(raw ? JSON.parse(raw) : null);
  } catch (error) {
    console.warn("无法读取书写范围校准。", error);
    state.calibrationRange = null;
  }
}

function persistCalibration() {
  try {
    if (state.calibrationRange) {
      window.localStorage.setItem(CALIBRATION_KEY, JSON.stringify(state.calibrationRange));
    } else {
      window.localStorage.removeItem(CALIBRATION_KEY);
    }
  } catch (error) {
    console.warn("无法保存书写范围校准。", error);
  }
}

function smoothingLabel(value) {
  if (value >= 76) return "平稳";
  if (value >= 36) return "适中";
  return "灵敏";
}

function penWidthLabel(value) {
  return ["细", "较细", "标准", "较粗", "粗", "很粗"][clamp(Math.round(value), 1, 6) - 1];
}

function pinchThresholdLabel(value) {
  if (value < 0.3) return "较严格";
  if (value < 0.44) return "标准";
  if (value < 0.54) return "较易落笔";
  return "很容易";
}

function releaseDelayLabel(value) {
  if (value < 110) return "干脆";
  if (value < 200) return "适中";
  return "更连贯";
}

function getInkWidth(canvasWidth) {
  return canvasWidth * (0.002 + settings.penWidth * 0.001);
}

function smoothingCutoff() {
  return 3.1 - (settings.smoothing / 100) * 2.3;
}

function syncSettingControls() {
  els.pinchThreshold.value = String(settings.threshold);
  els.pinchThresholdValue.textContent = pinchThresholdLabel(settings.threshold);
  els.pinchThreshold.setAttribute("aria-valuetext", pinchThresholdLabel(settings.threshold));
  els.releaseDelay.value = String(settings.releaseDelay);
  els.releaseDelayValue.textContent = `${releaseDelayLabel(settings.releaseDelay)} · ${settings.releaseDelay}ms`;
  els.releaseDelay.setAttribute(
    "aria-valuetext",
    `${releaseDelayLabel(settings.releaseDelay)}，${settings.releaseDelay} 毫秒`,
  );
  els.smoothing.value = String(settings.smoothing);
  els.smoothingValue.textContent = smoothingLabel(settings.smoothing);
  els.mapGain.value = String(Math.round(settings.mapGain * 100));
  els.mapGainValue.textContent = `${Math.round(settings.mapGain * 100)}%`;
  els.penWidth.value = String(settings.penWidth);
  els.penWidthValue.textContent = penWidthLabel(settings.penWidth);
  els.directionMode.value = settings.directionMode;
  els.mirrorToggle.checked = settings.mirror;
  els.cameraVideo.classList.toggle("mirrored", settings.mirror);
  updateDirectionLabels();
  updatePinchMark();
  updateCalibrationUi();
}

function updateDirectionLabels() {
  const invert = settings.directionMode === "invert";
  setText(els.directionText, invert ? "左右映射已反向" : "笔迹方向与预览一致");
  setText(
    els.directionHint,
    invert ? "（手动校准）" : settings.mirror ? "（镜像模式）" : "（原始方向）",
  );
  setText(els.directionTagText, settings.mirror ? "镜像预览" : "原始画面");
}

function updatePinchMark() {
  const pct = clamp((1 - settings.threshold / MAX_METER_RATIO) * 100, 2, 98);
  els.pinchMark.style.left = `${pct}%`;
}

/* -------------------------------------------------------------------------- */
/* Personal writing-range calibration                                         */
/* -------------------------------------------------------------------------- */

function canStartCalibration() {
  return (
    state.camera === "live" &&
    state.model === "ready" &&
    state.writing !== "writing" &&
    !state.calibrationActive
  );
}

function currentCalibrationStep() {
  return CALIBRATION_STEPS[Math.min(state.calibrationStep, CALIBRATION_STEPS.length - 1)];
}

function updateCalibrationUi() {
  const canCalibrate = canStartCalibration();
  const calibrated = Boolean(state.calibrationRange);

  if (els.calibrateBtn) {
    els.calibrateBtn.disabled = !canCalibrate;
    setText(
      els.calibrateBtnText,
      state.calibrationActive ? "校准中…" : calibrated ? "重做校准" : "校准范围",
    );
  }
  if (els.settingsCalibrateBtn) {
    els.settingsCalibrateBtn.disabled = !canCalibrate;
    els.settingsCalibrateBtn.textContent = calibrated ? "重做校准" : "校准书写范围";
  }
  if (els.resetCalibrationBtn) {
    els.resetCalibrationBtn.disabled = !calibrated || state.calibrationActive;
  }
  if (els.calibrationState) {
    els.calibrationState.textContent = state.calibrationActive
      ? "校准中"
      : calibrated
        ? "已校准"
        : "默认范围";
  }
  if (els.mirrorToggle) {
    els.mirrorToggle.disabled = state.calibrationActive;
  }

  if (!els.calibrationPanel) return;
  els.calibrationPanel.hidden = !state.calibrationActive;
  if (!state.calibrationActive) return;

  const step = currentCalibrationStep();
  if (els.calibrationProgress) {
    setText(els.calibrationProgress, `第 ${state.calibrationStep + 1} / 4 点 · ${step.shortName}`);
  }
  if (els.calibrationInstruction) {
    setText(
      els.calibrationInstruction,
      `将准备书写的食指移到舒适书写范围的${step.name}，保持自然姿势，然后点击「记录此点」。`,
    );
  }
  if (els.calibrationHint) {
    setText(
      els.calibrationHint,
      state.handDetected
        ? "四个角尽量形成一个约 12:5 的横向矩形；校准只调整映射，不会修改笔迹形状。"
        : state.camera === "live"
          ? "未检测到手，请将准备书写的整只手放入画面。"
          : "请先开启摄像头，再开始校准。",
    );
  }
  if (els.calibrationCaptureBtn) {
    els.calibrationCaptureBtn.disabled = !state.handDetected || !state.latestCursor;
  }
}

function startCalibration() {
  if (!canStartCalibration()) {
    if (state.camera !== "live") showToast("请先开启摄像头，再校准书写范围");
    else if (state.model !== "ready") showToast("手部识别尚未就绪，请稍候");
    else if (state.writing === "writing") showToast("请先结束当前书写，再校准书写范围");
    return;
  }

  state.calibrationActive = true;
  state.calibrationStep = 0;
  state.calibrationPoints = [];
  els.cameraFrame.classList.add("is-calibrating");
  updateCalibrationUi();
  syncUi();
  drawOverlay();
  showToast("开始校准：依次记录书写范围的左上、右上、左下、右下四个角", { duration: 4200 });
  announce("开始校准书写范围，请记录四个角");
  if (els.calibrationPanel) {
    els.calibrationPanel.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function finishCalibration() {
  const points = state.calibrationPoints;
  if (points.length < 4) return;

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  let minX = Math.min(...xs);
  let maxX = Math.max(...xs);
  let minY = Math.min(...ys);
  let maxY = Math.max(...ys);
  const padX = Math.max(0.008, (maxX - minX) * 0.02);
  const padY = Math.max(0.008, (maxY - minY) * 0.02);
  minX = clamp(minX - padX, 0.01, 0.99);
  maxX = clamp(maxX + padX, 0.01, 0.99);
  minY = clamp(minY - padY, 0.01, 0.99);
  maxY = clamp(maxY + padY, 0.01, 0.99);

  if (maxX - minX < 0.16 || maxY - minY < 0.1) {
    state.calibrationPoints = [];
    state.calibrationStep = 0;
    updateCalibrationUi();
    drawOverlay();
    showToast("校准范围太小，请重新记录并拉开四个角", { error: true, duration: 3600 });
    return;
  }

  state.calibrationRange = { minX, maxX, minY, maxY };
  persistCalibration();
  stopCalibration({ message: "书写范围已更新，现在可以按自己的姿势书写" });
}

function stopCalibration({ message = "", silent = false } = {}) {
  const wasActive = state.calibrationActive;
  state.calibrationActive = false;
  state.calibrationPoints = [];
  state.calibrationStep = 0;
  if (els.cameraFrame) els.cameraFrame.classList.remove("is-calibrating");
  updateCalibrationUi();
  syncUi();
  drawOverlay();
  if (wasActive && message) {
    showToast(message, { duration: 3000 });
    announce(message);
  } else if (wasActive && !silent) {
    showToast("已退出书写范围校准");
  }
}

function captureCalibrationPoint() {
  if (!state.calibrationActive || !state.handDetected || !state.latestCursor) {
    showToast("未检测到指尖，请将手放入画面后再记录");
    return;
  }

  const step = currentCalibrationStep();
  state.calibrationPoints.push({ ...state.latestCursor });
  state.calibrationStep += 1;
  drawOverlay();

  if (state.calibrationStep >= 4) {
    finishCalibration();
    return;
  }

  updateCalibrationUi();
  syncUi();
  showToast(`已记录${step.shortName}，请移动到${currentCalibrationStep().name}`);
  announce(`已记录${step.name}，请记录${currentCalibrationStep().name}`);
}

function abortCalibrationSilently() {
  if (!state.calibrationActive) return;
  state.calibrationActive = false;
  state.calibrationPoints = [];
  state.calibrationStep = 0;
  if (els.cameraFrame) els.cameraFrame.classList.remove("is-calibrating");
}

function skipCalibration() {
  stopCalibration({ message: "已跳过校准，继续使用当前书写范围" });
}

function resetCalibrationRange() {
  state.calibrationRange = null;
  persistCalibration();
  if (state.calibrationActive) {
    state.calibrationPoints = [];
    state.calibrationStep = 0;
    stopCalibration({ message: "已恢复默认书写范围" });
    return;
  }
  updateCalibrationUi();
  syncUi();
  drawOverlay();
  showToast("已恢复默认书写范围");
  announce("已恢复默认书写范围");
}

function mirrorCalibrationRange() {
  if (!state.calibrationRange) return;
  const current = state.calibrationRange;
  state.calibrationRange = {
    minX: 1 - current.maxX,
    maxX: 1 - current.minX,
    minY: current.minY,
    maxY: current.maxY,
  };
  persistCalibration();
}

/* -------------------------------------------------------------------------- */
/* One Euro smoothing                                                         */
/* -------------------------------------------------------------------------- */

class LowPassFilter {
  constructor() {
    this.value = null;
  }

  filter(input, alpha) {
    this.value = this.value === null ? input : this.value + alpha * (input - this.value);
    return this.value;
  }

  reset() {
    this.value = null;
  }
}

class OneEuroFilter {
  constructor({ minCutoff = 1.5, beta = 18, dCutoff = 1 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.lastTime = null;
    this.lastValue = 0;
    this.filteredDerivative = 0;
    this.valueFilter = new LowPassFilter();
    this.derivativeFilter = new LowPassFilter();
  }

  alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(value, timestamp) {
    if (!Number.isFinite(value) || !Number.isFinite(timestamp)) return value;

    if (this.lastTime === null) {
      this.lastTime = timestamp;
      this.lastValue = value;
      this.filteredDerivative = 0;
      this.valueFilter.filter(value, 1);
      return value;
    }

    const dt = clamp(timestamp - this.lastTime, 1 / 240, 0.2);
    const derivative = (value - this.lastValue) / dt;
    const dAlpha = this.alpha(this.dCutoff, dt);
    this.filteredDerivative = this.derivativeFilter.filter(derivative, dAlpha);

    const cutoff = this.minCutoff + this.beta * Math.abs(this.filteredDerivative);
    const alpha = this.alpha(cutoff, dt);
    const filtered = this.valueFilter.filter(value, alpha);

    this.lastTime = timestamp;
    this.lastValue = value;
    return filtered;
  }
}

function resetFingerFilters() {
  const cutoff = smoothingCutoff();
  state.filters = {
    x: new OneEuroFilter({ minCutoff: cutoff, beta: 26, dCutoff: 1 }),
    y: new OneEuroFilter({ minCutoff: cutoff, beta: 26, dCutoff: 1 }),
  };
}

function updateFilterSmoothing() {
  if (!state.filters) return;
  const cutoff = smoothingCutoff();
  state.filters.x.minCutoff = cutoff;
  state.filters.y.minCutoff = cutoff;
}

/* -------------------------------------------------------------------------- */
/* Canvas sizing and ink rendering                                            */
/* -------------------------------------------------------------------------- */

function fitOverlayCanvas() {
  const rect = els.overlayCanvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rect.width * dpr);
  const height = Math.round(rect.height * dpr);
  if (els.overlayCanvas.width !== width || els.overlayCanvas.height !== height) {
    els.overlayCanvas.width = width;
    els.overlayCanvas.height = height;
  }
  state.overlayW = rect.width;
  state.overlayH = rect.height;
  state.overlayDpr = dpr;
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  drawOverlay();
}

function fitSignatureCanvas() {
  const rect = els.signatureCanvas.getBoundingClientRect();
  if (rect.width < 2 || rect.height < 2) return;

  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const width = Math.round(rect.width * dpr);
  const height = Math.round(rect.height * dpr);
  if (els.signatureCanvas.width !== width || els.signatureCanvas.height !== height) {
    els.signatureCanvas.width = width;
    els.signatureCanvas.height = height;
  }
  state.signatureW = rect.width;
  state.signatureH = rect.height;
  signatureCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  renderSignature();
}

function renderSignature() {
  if (!signatureCtx || !state.signatureW || !state.signatureH) return;
  const width = state.signatureW;
  const height = state.signatureH;
  signatureCtx.clearRect(0, 0, width, height);
  signatureCtx.strokeStyle = "#17231f";
  signatureCtx.fillStyle = "#17231f";
  signatureCtx.lineCap = "round";
  signatureCtx.lineJoin = "round";

  const lineWidth = getInkWidth(width);
  for (const stroke of state.strokes) {
    drawInkStroke(signatureCtx, stroke, width, height, lineWidth);
  }
  if (state.currentStroke && state.currentStroke.length) {
    drawInkStroke(signatureCtx, state.currentStroke, width, height, lineWidth);
  }
}

function drawInkStroke(ctx, points, width, height, lineWidth) {
  if (!points || !points.length) return;
  ctx.lineWidth = lineWidth;

  if (points.length === 1) {
    ctx.beginPath();
    ctx.arc(points[0].x * width, points[0].y * height, lineWidth * 0.54, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  ctx.beginPath();
  ctx.moveTo(points[0].x * width, points[0].y * height);

  if (points.length === 2) {
    ctx.lineTo(points[1].x * width, points[1].y * height);
  } else {
    for (let index = 1; index < points.length - 1; index += 1) {
      const point = points[index];
      const next = points[index + 1];
      const midX = ((point.x + next.x) / 2) * width;
      const midY = ((point.y + next.y) / 2) * height;
      ctx.quadraticCurveTo(point.x * width, point.y * height, midX, midY);
    }
    const last = points[points.length - 1];
    ctx.lineTo(last.x * width, last.y * height);
  }

  ctx.stroke();
}

function getVideoDimensions() {
  return {
    width: els.cameraVideo.videoWidth || 1280,
    height: els.cameraVideo.videoHeight || 720,
  };
}

function getDefaultRoi() {
  const { width: videoWidth, height: videoHeight } = getVideoDimensions();
  const cameraAspect = videoWidth / videoHeight;
  const roiWidth = DEFAULT_ROI_X.maxX - DEFAULT_ROI_X.minX;
  // Fit the default region to the canvas aspect so comfortable hand movement
  // does not get stretched after mapping.
  const idealHeight = (roiWidth * cameraAspect) / TARGET_ASPECT;
  const roiHeight = clamp(idealHeight, 0.3, 0.74);
  const minY = clamp(0.5 - roiHeight / 2, 0.06, 0.94 - roiHeight);
  return {
    minX: DEFAULT_ROI_X.minX,
    maxX: DEFAULT_ROI_X.maxX,
    minY,
    maxY: minY + roiHeight,
  };
}

function getActiveRoi() {
  return state.calibrationRange
    ? { ...state.calibrationRange }
    : getDefaultRoi();
}

function getMappingGeometry() {
  const roi = getActiveRoi();
  const { width: videoWidth, height: videoHeight } = getVideoDimensions();
  const roiPixelWidth = Math.max(1, (roi.maxX - roi.minX) * videoWidth);
  const roiPixelHeight = Math.max(1, (roi.maxY - roi.minY) * videoHeight);
  const safePixelWidth = SAFE_AREA.width * EXPORT_WIDTH;
  const safePixelHeight = SAFE_AREA.height * EXPORT_HEIGHT;
  const scale = Math.min(safePixelWidth / roiPixelWidth, safePixelHeight / roiPixelHeight);
  const mappedWidth = roiPixelWidth * scale;
  const mappedHeight = roiPixelHeight * scale;
  const left = SAFE_AREA.left * EXPORT_WIDTH + (safePixelWidth - mappedWidth) / 2;
  const top = SAFE_AREA.top * EXPORT_HEIGHT + (safePixelHeight - mappedHeight) / 2;
  return { roi, left, top, mappedWidth, mappedHeight };
}

function mapToCanvas(perceivedX, perceivedY) {
  const geometry = getMappingGeometry();
  const { roi } = geometry;
  const normalizedX = clamp((perceivedX - roi.minX) / (roi.maxX - roi.minX), 0, 1);
  const normalizedY = clamp((perceivedY - roi.minY) / (roi.maxY - roi.minY), 0, 1);

  let canvasX = (geometry.left + normalizedX * geometry.mappedWidth) / EXPORT_WIDTH;
  let canvasY = (geometry.top + normalizedY * geometry.mappedHeight) / EXPORT_HEIGHT;

  // Independent movement gain: 100% uses the calibrated mapping directly.
  // Clamp to the safe area so the signature never touches the exported edge.
  const gain = settings.mapGain;
  canvasX = 0.5 + (canvasX - 0.5) * gain;
  canvasY = 0.5 + (canvasY - 0.5) * gain;
  canvasX = clamp(canvasX, SAFE_AREA.left, SAFE_AREA.left + SAFE_AREA.width);
  canvasY = clamp(canvasY, SAFE_AREA.top, SAFE_AREA.top + SAFE_AREA.height);

  if (settings.directionMode === "invert") {
    canvasX = 1 - canvasX;
    canvasX = clamp(canvasX, SAFE_AREA.left, SAFE_AREA.left + SAFE_AREA.width);
  }

  return { x: canvasX, y: canvasY };
}

function appendPoint(point) {
  if (!state.currentStroke || !point) return;
  const last = state.currentStroke[state.currentStroke.length - 1];
  const dx = point.x - last.x;
  const dy = point.y - last.y;
  if (dx * dx + dy * dy < 0.0000032) return;
  state.currentStroke.push(point);
  renderSignature();
}

function beginStroke() {
  if (state.writing !== "writing" || state.blockUntilRelease || state.currentStroke) return false;
  if (state.calibrationActive || performance.now() < state.drawBlockedUntil) return false;
  if (!state.latestCanvasPoint) return false;
  state.currentStroke = [{ ...state.latestCanvasPoint }];
  renderSignature();
  return true;
}

function endStroke(rawPoint = null) {
  if (!state.currentStroke) return false;

  if (rawPoint && state.currentStroke.length) {
    const last = state.currentStroke[state.currentStroke.length - 1];
    const dx = rawPoint.x - last.x;
    const dy = rawPoint.y - last.y;
    const distanceSquared = dx * dx + dy * dy;
    // Close the visible gap left by smoothing lag, without accepting a wild jump.
    if (distanceSquared > 0.0000016 && distanceSquared < 0.0006) {
      state.currentStroke.push({ ...rawPoint });
    }
  }

  if (state.currentStroke.length) {
    state.strokes.push(state.currentStroke);
  }
  state.currentStroke = null;
  renderSignature();
  return true;
}

function clearInk() {
  state.strokes = [];
  state.currentStroke = null;
  renderSignature();
}

/* -------------------------------------------------------------------------- */
/* Status and UI                                                              */
/* -------------------------------------------------------------------------- */

function syncUi() {
  const live = state.camera === "live";
  const ready = live && state.model === "ready";
  const ink = hasInk();

  let cameraTone = "idle";
  let cameraLabel = "摄像头未开启";
  if (state.camera === "requesting") {
    cameraTone = "requesting";
    cameraLabel = "等待摄像头授权";
  } else if (state.camera === "error") {
    cameraTone = "error";
    cameraLabel = "摄像头不可用";
  } else if (live) {
    if (state.model === "error") {
      cameraTone = "error";
      cameraLabel = "识别模型异常";
    } else if (state.model === "ready") {
      cameraTone = "live";
      cameraLabel = "摄像头运行中";
    } else {
      cameraTone = "requesting";
      cameraLabel = state.model === "loading" ? "识别模型加载中" : "正在准备识别";
    }
  }
  setPill(els.cameraPill, els.cameraPillText, cameraTone, cameraLabel);

  els.cameraFrame.classList.toggle("is-live", live);
  els.cameraFrame.classList.toggle(
    "is-loading",
    state.camera === "requesting" || (live && (state.model === "loading" || state.model === "idle")),
  );
  els.cameraPlaceholder.classList.toggle("is-hidden", live);

  if (state.camera === "requesting") {
    setText(els.placeholderTitle, "正在等待摄像头授权…");
    setText(els.placeholderDetail, "请在浏览器弹窗中选择「允许」，画面不会离开本机");
  } else if (state.camera === "error") {
    setText(els.placeholderTitle, "摄像头暂不可用");
    setText(els.placeholderDetail, "请检查浏览器权限与摄像头连接，然后重新尝试");
  } else {
    setText(els.placeholderTitle, "摄像头尚未开启");
    setText(els.placeholderDetail, "点击「开启摄像头」，并允许浏览器访问");
  }

  setText(
    els.liveTagText,
    live ? (state.model === "ready" ? "LIVE" : state.model === "error" ? "ERROR" : "LOADING") : state.camera === "requesting" ? "WAIT" : "STANDBY",
  );
  els.liveTag.dataset.live = String(live);

  els.cameraStartBtn.hidden = live;
  els.cameraStartBtn.disabled = state.camera === "requesting";
  setText(
    els.cameraStartText,
    state.camera === "requesting"
      ? "正在请求权限…"
      : state.camera === "error"
        ? "重试摄像头"
        : "开启摄像头",
  );
  els.cameraStopBtn.hidden = !live;

  const handLabel = !live
    ? "等待画面"
    : !ready
      ? state.model === "error"
        ? "识别不可用"
        : "模型加载中"
      : state.handDetected
        ? "已检测到手"
        : "未检测到手";
  setText(els.handStateText, handLabel);
  els.handStateText.dataset.active = String(ready && state.handDetected);

  setText(
    els.cameraStateText,
    state.camera === "requesting"
      ? "授权中"
      : state.camera === "error"
        ? "异常"
        : live
          ? "运行中"
          : "关闭",
  );

  setText(
    els.writingStateText,
    state.writing === "writing" ? "正在书写" : state.writing === "finished" ? "已结束" : "未开始",
  );
  let pinchLabel = "—";
  if (ready && state.handDetected) {
    if (state.writing === "writing" && performance.now() < state.drawBlockedUntil) {
      pinchLabel = "重新定位";
    } else if (state.writing === "writing" && state.blockUntilRelease && state.pinching) {
      pinchLabel = "待松开";
    } else if (state.writing === "writing") {
      pinchLabel = state.pinching ? "已落笔" : "已抬笔";
    } else {
      pinchLabel = state.pinching ? "已捏合" : "未捏合";
    }
  }
  setText(els.pinchStateText, pinchLabel);
  els.pinchStateText.dataset.active = String(
    ready && state.handDetected && state.pinching && !state.blockUntilRelease,
  );

  const writingTone =
    state.writing === "writing" ? "writing" : state.writing === "finished" ? "finished" : "idle";
  setPill(
    els.writingPill,
    els.writingPillText,
    writingTone,
    state.writing === "writing" ? "正在书写" : state.writing === "finished" ? "已结束" : "未开始",
  );

  const handPillText = !live
    ? "未检测到手"
    : !ready
      ? state.model === "error"
        ? "识别不可用"
        : "正在加载手部模型"
      : state.handDetected
        ? state.pinching
          ? state.writing === "writing" && state.blockUntilRelease
            ? "已检测到手 · 请先松开"
            : state.writing === "writing"
              ? "已检测到手 · 落笔中"
              : "已检测到手 · 捏合中"
          : "已检测到手"
        : "未检测到手";
  setText(els.handPillText, handPillText);
  els.handPill.dataset.detected = String(ready && state.handDetected);
  els.handPill.dataset.pinching = String(ready && state.handDetected && state.pinching);

  els.toggleWritingBtn.textContent =
    state.writing === "writing" ? "结束书写" : state.writing === "finished" ? "书写已结束" : "开始书写";
  els.toggleWritingBtn.classList.toggle("is-ending", state.writing === "writing");
  els.toggleWritingBtn.disabled =
    state.calibrationActive
      ? true
      : state.writing === "writing"
        ? false
        : !ready || state.writing === "finished";

  els.rewriteBtn.hidden = state.writing !== "finished";
  els.rewriteBtn.disabled = state.writing !== "finished" || !ready || state.calibrationActive;

  const canExport = state.writing === "finished" && ink && !state.exporting;
  els.clearBtn.disabled = state.writing !== "finished" || !ink;
  els.undoBtn.disabled = state.writing !== "finished" || state.strokes.length === 0;
  els.exportBtn.disabled = !canExport;
  els.exportOptions.disabled = !(state.writing === "finished" && ink);
  setText(els.exportBtnText, state.exporting ? "正在生成…" : "预览导出");

  setText(els.inkCount, ink ? `${state.strokes.length + (state.currentStroke && state.currentStroke.length ? 1 : 0)} 笔` : "尚无笔迹");
  els.signaturePanel.classList.toggle("is-writing", state.writing === "writing");
  els.signatureEmpty.classList.toggle("is-hidden", ink || state.writing === "writing");
  els.frozenBadge.hidden = !(state.writing === "finished" && ink);

  els.writingTip.dataset.tone =
    state.writing === "finished" ? "done" : state.writing === "writing" ? "active" : "neutral";
  setText(els.writingTipText, getWritingTip({ live, ready }));
  setText(els.frameHint, getFrameHint({ live, ready }));
  updatePinchMeter();
  updateSignaturePenCursor();
  updateCalibrationUi();
}

function getWritingTip({ live, ready }) {
  if (state.calibrationActive) {
    if (!live || !ready) return "校准需要摄像头和手部识别就绪。";
    if (!state.handDetected) return "校准中：请将准备书写的整只手放入画面，再移动到目标角落。";
    return `校准中：将食指移到${currentCalibrationStep().name}，保持自然姿势，然后点击「记录此点」。`;
  }

  if (state.writing === "finished") {
    return hasInk()
      ? "笔迹已冻结。可预览导出、撤销最后一笔，或点击「重新书写」再来一次。"
      : "书写已结束，本次没有留下笔迹。可点击「重新书写」再试一次。";
  }

  if (state.writing === "writing" && !state.handDetected) {
    return "未检测到手部，当前笔画已结束。请将手重新放入画面，等待笔尖稳定后再捏合起笔。";
  }

  if (state.writing === "writing" && performance.now() < state.drawBlockedUntil) {
    return state.stabilityReason === "jump"
      ? "检测到笔尖跳变，已自动抬笔。正在等待轨迹稳定，请保持捏合姿势。"
      : "手部刚重新入镜，正在等待轨迹稳定。稍后会自动继续，不会连接丢失前的笔迹。";
  }

  if (state.writing === "writing") {
    if (state.blockUntilRelease && state.pinching) {
      return "请先松开拇指与食指，再重新捏合开始书写。";
    }
    if (state.pinching) {
      return "已落笔。画布中的圆点就是笔尖，按自然速度移动食指即可；短暂抖动不会轻易断笔。";
    }
    return "书写中。画布中的空心圆是笔尖位置；捏合拇指与食指即可开始一笔。";
  }

  if (!live) {
    return "首次使用：开启摄像头并进入书写模式后，捏合拇指与食指落笔，松开换笔；完成后点击「结束书写」。";
  }
  if (!ready) {
    return state.model === "error"
      ? "手部识别不可用。请先按上方提示重新加载识别模型。"
      : "正在加载手部识别模型，完成后即可开始书写。";
  }
  if (state.handDetected) {
    return "手部已就绪。点击「开始书写」，然后捏合落笔。";
  }
  return "点击「开始书写」后，将手完整放入画面，再捏合拇指与食指落笔。";
}

function getFrameHint({ live, ready }) {
  if (state.calibrationActive) {
    if (!state.handDetected) return "校准中 · 请将书写手放入画面";
    return `校准中 · 移动到${currentCalibrationStep().name}后记录`;
  }
  if (state.camera === "requesting") return "请在浏览器权限提示中选择「允许」";
  if (state.camera === "error") return "请检查摄像头连接与浏览器权限后重试";
  if (!live) return "手部入镜后，画面中会显示骨架";

  if (!ready) {
    return state.model === "error" ? "识别模型不可用，请按上方提示重试" : "正在加载手部识别模型，请稍候…";
  }

  if (!state.handDetected) {
    return state.writing === "writing"
      ? "未检测到手，已自动抬笔；请将手重新放入画面"
      : "请将一只手完整放入画面，掌心朝向镜头";
  }

  if (state.writing === "writing") {
    if (performance.now() < state.drawBlockedUntil) {
      return state.stabilityReason === "jump"
        ? "检测到异常跳点，已自动抬笔，正在重新稳定"
        : "手部刚重新入镜，正在等待指尖稳定";
    }
    if (state.blockUntilRelease && state.pinching) return "请先松开手指，再重新捏合开始书写";
    if (state.pinching) return "正在落笔 · 画布中的圆点即笔尖";
    return "书写中 · 空心圆为笔尖位置";
  }
  if (state.writing === "finished") return "书写已结束 · 可重新书写或导出";
  return "已识别手部 · 可点击「开始书写」";
}

function updatePinchMeter() {
  if (state.pinchRatio === null || !state.handDetected) {
    els.pinchBar.style.width = "0%";
    return;
  }
  const pct = clamp((1 - state.pinchRatio / MAX_METER_RATIO) * 100, 0, 100);
  els.pinchBar.style.width = `${pct}%`;
}

function updateSignaturePenCursor() {
  if (!els.penCursor) return;
  const canShow =
    state.writing === "writing" &&
    !state.calibrationActive &&
    state.camera === "live" &&
    state.model === "ready" &&
    state.handDetected &&
    state.latestCanvasPoint &&
    performance.now() >= state.drawBlockedUntil;

  if (!canShow) {
    els.penCursor.classList.remove("is-visible");
    return;
  }

  els.penCursor.style.left = `${(state.latestCanvasPoint.x * 100).toFixed(3)}%`;
  els.penCursor.style.top = `${(state.latestCanvasPoint.y * 100).toFixed(3)}%`;
  els.penCursor.dataset.down = String(Boolean(state.pinching && !state.blockUntilRelease));
  els.penCursor.classList.add("is-visible");
}

/* -------------------------------------------------------------------------- */
/* Camera                                                                     */
/* -------------------------------------------------------------------------- */

function describeCameraError(error) {
  if (!window.isSecureContext) {
    return "当前页面不是安全上下文。请通过 http://localhost 或 HTTPS 访问后再授权摄像头。";
  }
  switch (error && error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "摄像头权限被拒绝。请点击地址栏左侧的摄像头图标，将本站设为「允许」后点击「重试摄像头」。";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "未找到可用的摄像头。请连接摄像头，或检查系统是否识别到拍摄设备后重试。";
    case "NotReadableError":
    case "TrackStartError":
      return "摄像头暂时无法读取，可能正被其他应用占用。请关闭占用摄像头的应用后重试。";
    case "OverconstrainedError":
      return "摄像头不支持当前分辨率要求。请更换摄像头或刷新页面后重试。";
    case "AbortError":
      return "摄像头启动被中断。请重新点击「重试摄像头」。";
    default:
      return "摄像头启动失败。请检查浏览器权限和设备连接后重试。";
  }
}

function releaseStream() {
  const stream = state.stream;
  state.stream = null;
  if (stream) {
    for (const track of stream.getTracks()) {
      try {
        track.stop();
      } catch (error) {
        console.warn("关闭摄像头轨道失败", error);
      }
    }
  }
  if (els.cameraVideo.srcObject) {
    els.cameraVideo.srcObject = null;
  }
}

function waitForVideoReady(video) {
  return new Promise((resolve, reject) => {
    if (video.readyState >= 1) {
      resolve();
      return;
    }

    let timer = 0;
    const cleanup = () => {
      window.clearTimeout(timer);
      video.removeEventListener("loadedmetadata", onLoaded);
      video.removeEventListener("error", onError);
    };
    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("摄像头画面无法播放"));
    };

    timer = window.setTimeout(() => {
      cleanup();
      reject(new Error("等待摄像头画面超时"));
    }, 12000);

    video.addEventListener("loadedmetadata", onLoaded, { once: true });
    video.addEventListener("error", onError, { once: true });
  });
}

async function startCamera() {
  if (state.camera === "requesting" || state.camera === "live") return;

  hideAlert("camera");
  hideAlert("model");

  if (!window.isSecureContext) {
    const message = "当前页面不是安全上下文。请通过 http://localhost 或 HTTPS 访问，浏览器才允许使用摄像头。";
    state.camera = "error";
    showAlert("camera", message, "重试摄像头", startCamera);
    syncUi();
    showToast("无法访问摄像头：请使用 localhost 或 HTTPS", { error: true, duration: 4200 });
    announce(message);
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const message = "当前浏览器不支持摄像头访问。请使用最新版 Chrome、Edge 或 Safari 打开本页面。";
    state.camera = "error";
    showAlert("camera", message, "重试摄像头", startCamera);
    syncUi();
    showToast("当前浏览器不支持摄像头访问", { error: true });
    return;
  }

  state.camera = "requesting";
  state.handDetected = false;
  state.pinching = false;
  state.lockedHandedness = null;
  state.lastWristPerceived = null;
  state.pinchRatio = null;
  state.pinchRatioSmoothed = null;
  state.pinchFrames = 0;
  state.releaseFrames = 0;
  state.releaseSince = null;
  state.latestRawCanvasPoint = null;
  syncUi();
  announce("正在请求摄像头权限");

  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
      audio: false,
    });

    if (state.camera !== "requesting") {
      for (const track of stream.getTracks()) track.stop();
      return;
    }

    state.stream = stream;
    const videoTrack = stream.getVideoTracks()[0];
    if (videoTrack) {
      videoTrack.addEventListener(
        "ended",
        () => {
          if (state.stream !== stream) return;
          handleUnexpectedCameraEnd();
        },
        { once: true },
      );
    }

    els.cameraVideo.srcObject = stream;
    await waitForVideoReady(els.cameraVideo);
    await els.cameraVideo.play();

    state.camera = "live";
    state.lastVideoTime = -1;
    hideAlert("camera");
    resetFingerFilters();
    fitOverlayCanvas();
    syncUi();
    announce("摄像头已开启，正在加载手部识别模型");

    await ensureModel();
    if (state.camera === "live" && state.model === "ready") {
      startTracking();
    }
  } catch (error) {
    console.error("摄像头启动失败", error);
    if (stream) {
      for (const track of stream.getTracks()) {
        try {
          track.stop();
        } catch (stopError) {
          console.warn(stopError);
        }
      }
    }
    if (state.stream === stream) state.stream = null;
    els.cameraVideo.srcObject = null;

    const message = error && error.name === undefined
      ? "无法播放摄像头画面。请确认摄像头未被占用，并尝试重新授权。"
      : error && error.message && /摄像头画面/.test(error.message)
        ? `${error.message}。请关闭占用摄像头的应用后重试。`
        : describeCameraError(error);
    state.camera = "error";
    state.handDetected = false;
    state.pinching = false;
    state.pinchRatio = null;
    state.pinchRatioSmoothed = null;
    state.pinchFrames = 0;
    state.releaseFrames = 0;
    state.releaseSince = null;
    state.latestRawCanvasPoint = null;
    stopTracking();
    drawOverlay();
    syncUi();
    showAlert("camera", message, "重试摄像头", startCamera);
    showToast("摄像头未能开启，请按提示重试", { error: true, duration: 4200 });
    announce(message);
  }
}

function stopCamera({ silent = false } = {}) {
  const hadCamera = state.camera === "live" || state.camera === "requesting";
  const wasWriting = state.writing === "writing";

  stopTracking();
  releaseStream();
  abortCalibrationSilently();
  state.camera = "idle";
  state.handDetected = false;
  state.pinching = false;
  state.lockedHandedness = null;
  state.lastWristPerceived = null;
  state.pinchRatio = null;
  state.pinchRatioSmoothed = null;
  state.pinchFrames = 0;
  state.releaseFrames = 0;
  state.releaseSince = null;
  state.blockUntilRelease = false;
  state.drawBlockedUntil = 0;
  state.stabilityReason = null;
  state.latestLandmarks = null;
  state.latestCursor = null;
  state.latestCanvasPoint = null;
  state.latestRawCanvasPoint = null;
  endStroke();
  if (wasWriting) state.writing = "finished";
  resetFingerFilters();
  hideAlert("camera");
  drawOverlay();
  syncUi();

  if (hadCamera && !silent) {
    showToast(wasWriting ? "摄像头已关闭，当前签名已冻结" : "摄像头已关闭");
    announce(wasWriting ? "摄像头已关闭，当前签名已冻结" : "摄像头已关闭");
  }
  if (wasWriting) {
    els.signatureEmpty.classList.toggle("is-hidden", hasInk());
  }
}

function handleUnexpectedCameraEnd() {
  if (state.camera !== "live") return;
  const wasWriting = state.writing === "writing";

  stopTracking();
  releaseStream();
  abortCalibrationSilently();
  state.camera = "error";
  state.handDetected = false;
  state.pinching = false;
  state.lockedHandedness = null;
  state.lastWristPerceived = null;
  state.pinchRatio = null;
  state.pinchRatioSmoothed = null;
  state.pinchFrames = 0;
  state.releaseFrames = 0;
  state.releaseSince = null;
  state.blockUntilRelease = false;
  state.drawBlockedUntil = 0;
  state.stabilityReason = null;
  state.latestLandmarks = null;
  state.latestCursor = null;
  state.latestCanvasPoint = null;
  state.latestRawCanvasPoint = null;
  endStroke();
  if (wasWriting) state.writing = "finished";
  resetFingerFilters();
  drawOverlay();
  syncUi();

  const message = "摄像头连接已断开。请检查设备连接，然后点击「重试摄像头」。";
  showAlert("camera", message, "重试摄像头", startCamera);
  showToast(wasWriting ? "摄像头断开，签名已自动冻结" : "摄像头连接已断开", {
    error: true,
    duration: 4200,
  });
  announce(wasWriting ? "摄像头连接已断开，当前签名已冻结" : message);
}

/* -------------------------------------------------------------------------- */
/* Hand Landmarker loading and frame loop                                     */
/* -------------------------------------------------------------------------- */

async function ensureModel() {
  if (state.model === "ready" && state.landmarker) return;
  if (state.modelPromise) return state.modelPromise;

  state.model = "loading";
  state.modelPromise = (async () => {
    try {
      const visionModule = await import("./assets/vendor/mediapipe/vision_bundle.mjs");
      if (!state.vision) {
        state.vision = await visionModule.FilesetResolver.forVisionTasks(WASM_PATH);
      }

      const baseOptions = {
        modelAssetPath: MODEL_PATH,
        delegate: "GPU",
      };
      const options = {
        baseOptions,
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      };

      let landmarker;
      try {
        landmarker = await visionModule.HandLandmarker.createFromOptions(state.vision, options);
      } catch (gpuError) {
        console.warn("GPU 手部识别初始化失败，回退到 CPU。", gpuError);
        landmarker = await visionModule.HandLandmarker.createFromOptions(state.vision, {
          ...options,
          baseOptions: { ...baseOptions, delegate: "CPU" },
        });
      }

      state.landmarker = landmarker;
      state.model = "ready";
      hideAlert("model");
      syncUi();
      announce("手部识别已就绪");
      showToast("手部识别已就绪，可以开始书写", { duration: 2600 });
      if (state.camera === "live") startTracking();
    } catch (error) {
      console.error("手部识别模型加载失败", error);
      state.model = "error";
      state.landmarker = null;
      state.vision = null;
      state.modelPromise = null;
      stopTracking();
      abortCalibrationSilently();
      state.lockedHandedness = null;
      state.lastWristPerceived = null;
      if (state.writing === "writing") {
        endStroke();
        state.writing = "finished";
      }
      state.handDetected = false;
      state.pinching = false;
      state.pinchRatio = null;
      state.pinchRatioSmoothed = null;
      state.pinchFrames = 0;
      state.releaseFrames = 0;
      state.releaseSince = null;
      state.drawBlockedUntil = 0;
      state.stabilityReason = null;
      state.latestLandmarks = null;
      state.latestCursor = null;
      state.latestCanvasPoint = null;
      state.latestRawCanvasPoint = null;
      drawOverlay();
      syncUi();
      showAlert(
        "model",
        "手部识别模型加载失败。请确认通过本地服务器打开页面且 assets 目录完整，然后点击「重新加载模型」。",
        "重新加载模型",
        retryModel,
      );
      showToast("手部识别模型加载失败，请重试", { error: true, duration: 4200 });
      announce("手部识别模型加载失败，请按提示重试");
    } finally {
      state.modelPromise = null;
      syncUi();
    }
  })();

  return state.modelPromise;
}

function retryModel() {
  if (state.model === "loading" || state.modelPromise) return;
  hideAlert("model");
  if (state.landmarker) {
    try {
      state.landmarker.close();
    } catch (error) {
      console.warn("关闭旧的识别实例失败", error);
    }
    state.landmarker = null;
  }
  ensureModel();
}

function startTracking() {
  if (state.rafId !== null) return;
  if (state.camera !== "live" || state.model !== "ready" || !state.landmarker) return;

  state.lastVideoTime = -1;
  const tick = (now) => {
    state.rafId = null;
    if (state.camera !== "live" || state.model !== "ready" || !state.landmarker) return;

    state.rafId = requestAnimationFrame(tick);

    const video = els.cameraVideo;
    if (video.readyState < 2 || !video.videoWidth) return;
    if (video.currentTime === state.lastVideoTime) return;
    state.lastVideoTime = video.currentTime;

    let result;
    try {
      result = state.landmarker.detectForVideo(video, now);
    } catch (error) {
      console.error("手部识别运行失败", error);
      handleTrackingFailure();
      return;
    }

    processResult(result, now);
  };

  state.rafId = requestAnimationFrame(tick);
}

function stopTracking() {
  if (state.rafId !== null) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
}

function handleTrackingFailure() {
  stopTracking();
  abortCalibrationSilently();
  state.lockedHandedness = null;
  state.lastWristPerceived = null;
  state.model = "error";
  if (state.writing === "writing") {
    endStroke();
    state.writing = "finished";
  }
  state.handDetected = false;
  state.pinching = false;
  state.pinchRatio = null;
  state.pinchRatioSmoothed = null;
  state.pinchFrames = 0;
  state.releaseFrames = 0;
  state.releaseSince = null;
  state.drawBlockedUntil = 0;
  state.stabilityReason = null;
  state.latestLandmarks = null;
  state.latestCursor = null;
  state.latestCanvasPoint = null;
  state.latestRawCanvasPoint = null;
  if (state.landmarker) {
    try {
      state.landmarker.close();
    } catch (error) {
      console.warn(error);
    }
    state.landmarker = null;
  }
  drawOverlay();
  syncUi();
  showAlert(
    "model",
    "手部识别运行中断。请重新加载识别模型；若反复出现，请刷新页面后再试。",
    "重新加载模型",
    retryModel,
  );
  showToast("手部识别中断，请重新加载模型", { error: true, duration: 4200 });
  announce("手部识别中断，请按提示重新加载模型");
}

/* -------------------------------------------------------------------------- */
/* Gesture processing                                                         */
/* -------------------------------------------------------------------------- */

function getHandedness(result, index) {
  const category = result && result.handednesses && result.handednesses[index]?.[0];
  return {
    label: category?.categoryName || null,
    score: Number(category?.score) || 0,
  };
}

function measurePinchRatio(landmarks) {
  const palmScale = Math.max(
    Math.hypot(landmarks[0].x - landmarks[9].x, landmarks[0].y - landmarks[9].y),
    Math.hypot(landmarks[5].x - landmarks[17].x, landmarks[5].y - landmarks[17].y),
    0.035,
  );
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  return Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y) / palmScale;
}

function selectPrimaryHand(result) {
  const hands = result && result.landmarks ? result.landmarks : [];
  if (!hands.length) return null;

  const candidates = hands.map((landmarks, index) => {
    const handedness = getHandedness(result, index);
    const wrist = landmarks[0];
    const perceivedWrist = {
      x: settings.mirror ? 1 - wrist.x : wrist.x,
      y: wrist.y,
    };
    const previous = state.lastWristPerceived;
    const wristDistance = previous
      ? Math.hypot(perceivedWrist.x - previous.x, perceivedWrist.y - previous.y)
      : 0;
    return {
      landmarks,
      label: handedness.label,
      score: handedness.score,
      wrist: perceivedWrist,
      wristDistance,
      pinchRatio: measurePinchRatio(landmarks),
    };
  });

  let pool = candidates;
  if (state.lockedHandedness) {
    const matching = candidates.filter((candidate) => candidate.label === state.lockedHandedness);
    if (matching.length) {
      pool = matching;
    } else if (candidates.length > 1) {
      // Do not jump to a second hand when the selected hand disappears.
      const nearest = [...candidates].sort((a, b) => a.wristDistance - b.wristDistance)[0];
      if (!state.lastWristPerceived || nearest.wristDistance > 0.28) return null;
      pool = [nearest];
    }
  }

  let selected;
  if (!state.lockedHandedness) {
    // First acquisition: prefer the hand already pinching, otherwise the strongest detection.
    const pinched = pool.filter((candidate) => candidate.pinchRatio <= settings.threshold);
    selected = (pinched.length ? pinched : pool).sort((a, b) => {
      if (pinched.length) return a.pinchRatio - b.pinchRatio;
      return b.score - a.score;
    })[0];
    state.lockedHandedness = selected.label || null;
  } else if (pool.length > 1) {
    selected = [...pool].sort((a, b) => {
      if (state.lastWristPerceived) return a.wristDistance - b.wristDistance;
      return b.score - a.score;
    })[0];
  } else {
    selected = pool[0];
  }

  if (!selected) return null;
  state.lastWristPerceived = selected.wrist;
  return selected;
}

function processResult(result, now) {
  const selected = selectPrimaryHand(result);
  const landmarks = selected ? selected.landmarks : null;
  if (!landmarks || landmarks.length < 21) {
    handleNoHand(now);
    drawOverlay();
    return;
  }

  const wasDetected = state.handDetected;
  state.handDetected = true;
  state.latestLandmarks = landmarks;

  const rawRatio = measurePinchRatio(landmarks);
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  state.pinchRatioSmoothed =
    state.pinchRatioSmoothed === null
      ? rawRatio
      : state.pinchRatioSmoothed + (rawRatio - state.pinchRatioSmoothed) * PINCH_RATIO_SMOOTHING;
  state.pinchRatio = state.pinchRatioSmoothed;

  const perceivedX = settings.mirror ? 1 - indexTip.x : indexTip.x;
  const perceivedY = indexTip.y;

  if (!state.filters) resetFingerFilters();
  const seconds = now / 1000;
  const smoothedX = clamp(state.filters.x.filter(perceivedX, seconds), 0, 1);
  const smoothedY = clamp(state.filters.y.filter(perceivedY, seconds), 0, 1);
  state.latestCursor = { x: smoothedX, y: smoothedY };
  state.latestCanvasPoint = mapToCanvas(smoothedX, smoothedY);
  state.latestRawCanvasPoint = mapToCanvas(perceivedX, perceivedY);

  let stabilityChanged = false;
  if (!wasDetected && state.drawBlockedUntil === Number.POSITIVE_INFINITY) {
    // Reacquired after a dropped hand: wait briefly before accepting a new stroke.
    state.drawBlockedUntil = now + STABILITY_WAIT_MS;
    state.stabilityReason = "reacquire";
    stabilityChanged = true;
  } else if (
    state.stabilityReason &&
    state.drawBlockedUntil !== Number.POSITIVE_INFINITY &&
    now >= state.drawBlockedUntil
  ) {
    state.drawBlockedUntil = 0;
    state.stabilityReason = null;
    stabilityChanged = true;
  }

  const pinchChanged = updatePinchState(state.pinchRatio, now);

  const canDraw = now >= state.drawBlockedUntil;
  if (
    canDraw &&
    state.writing === "writing" &&
    state.pinching &&
    !state.blockUntilRelease &&
    state.releaseSince === null
  ) {
    if (!state.currentStroke) {
      beginStroke();
    }
    if (state.currentStroke) {
      const lastPoint = state.currentStroke[state.currentStroke.length - 1];
      const jump = Math.hypot(
        (state.latestCanvasPoint.x - lastPoint.x) * EXPORT_WIDTH,
        (state.latestCanvasPoint.y - lastPoint.y) * EXPORT_HEIGHT,
      );
      if (jump > MAX_POINT_JUMP) {
        endStroke();
        state.drawBlockedUntil = now + STABILITY_WAIT_MS;
        state.stabilityReason = "jump";
        stabilityChanged = true;
        if (now - state.lastJumpNoticeAt > 1800) {
          state.lastJumpNoticeAt = now;
          showToast("检测到指尖跳变，已自动抬笔并等待重新稳定");
          announce("检测到指尖跳变，已自动抬笔");
        }
      } else {
        appendPoint(state.latestCanvasPoint);
      }
    }
  }

  updatePinchMeter();
  updateSignaturePenCursor();
  drawOverlay();

  if (!wasDetected || pinchChanged || stabilityChanged) {
    syncUi();
  }

  if (!wasDetected) {
    announce("已检测到手部");
  }
}

function updatePinchState(ratio, now) {
  const enterThreshold = settings.threshold;
  const releaseThreshold = Math.min(
    0.95,
    enterThreshold + Math.max(MIN_PINCH_RELEASE_GAP, enterThreshold * 0.45),
  );
  let changed = false;

  if (!state.pinching) {
    state.releaseSince = null;
    state.releaseFrames = 0;
    if (ratio <= enterThreshold) {
      state.pinchFrames += 1;
      if (state.pinchFrames >= 2) {
        state.pinching = true;
        state.pinchFrames = 0;
        state.releaseFrames = 0;
        state.releaseSince = null;
        changed = true;
        if (state.writing === "writing" && !state.blockUntilRelease) {
          beginStroke();
        }
      }
    } else {
      state.pinchFrames = 0;
    }
  } else if (ratio >= releaseThreshold) {
    if (state.releaseSince === null) state.releaseSince = now;
    state.releaseFrames += 1;
    // A short pinch-signal dip during fast handwriting should not break the stroke.
    if (state.releaseFrames >= 2 && now - state.releaseSince >= settings.releaseDelay) {
      state.pinching = false;
      state.releaseSince = null;
      state.releaseFrames = 0;
      state.pinchFrames = 0;
      changed = true;
      endStroke(state.latestRawCanvasPoint);
      if (state.blockUntilRelease) {
        state.blockUntilRelease = false;
        if (state.writing === "writing") {
          showToast("已松开手指，现在可以捏合开始书写");
        }
      }
    }
  } else {
    state.releaseSince = null;
    state.releaseFrames = 0;
  }

  return changed;
}

function handleNoHand() {
  const wasDetected = state.handDetected;
  const hadActiveStroke = Boolean(state.currentStroke);

  state.latestLandmarks = null;
  state.latestCursor = null;
  state.latestCanvasPoint = null;
  state.latestRawCanvasPoint = null;
  state.pinchRatio = null;
  state.pinchRatioSmoothed = null;

  if (!wasDetected) {
    updatePinchMeter();
    return;
  }

  endStroke();
  state.handDetected = false;
  state.pinching = false;
  state.pinchFrames = 0;
  state.releaseFrames = 0;
  state.releaseSince = null;
  state.blockUntilRelease = false;
  state.drawBlockedUntil = Number.POSITIVE_INFINITY;
  state.stabilityReason = state.writing === "writing" ? "reacquire" : null;
  state.lastWristPerceived = null;
  resetFingerFilters();
  updatePinchMeter();
  syncUi();

  if (hadActiveStroke) {
    showToast("手部离开画面，已自动抬笔");
    announce(state.writing === "writing" ? "手部离开画面，已自动抬笔" : "手部离开画面");
  } else {
    announce("未检测到手部");
  }
}

/* -------------------------------------------------------------------------- */
/* Writing actions                                                            */
/* -------------------------------------------------------------------------- */

function canBeginWriting() {
  return state.camera === "live" && state.model === "ready";
}

function startWriting({ showFeedback = true } = {}) {
  if (state.writing === "writing" || !canBeginWriting() || state.calibrationActive) return false;

  state.writing = "writing";
  state.blockUntilRelease =
    state.pinching || (state.pinchRatio !== null && state.pinchRatio <= settings.threshold);
  state.pinchFrames = 0;
  state.releaseFrames = 0;
  state.releaseSince = null;
  state.drawBlockedUntil = 0;
  state.stabilityReason = null;
  state.currentStroke = null;
  resetFingerFilters();
  renderSignature();
  syncUi();

  if (showFeedback) {
    const message = state.blockUntilRelease
      ? "书写模式已开始。请先松开手指，再捏合开始落笔。"
      : "书写模式已开始。捏合拇指与食指落笔，松开换笔。";
    showToast(message, { duration: 3200 });
    announce(message);
  }
  return true;
}

function finishWriting({ showFeedback = true } = {}) {
  if (state.writing !== "writing") return false;

  endStroke(state.latestRawCanvasPoint);
  state.writing = "finished";
  state.blockUntilRelease = false;
  renderSignature();
  syncUi();

  if (showFeedback) {
    const message = hasInk()
      ? "书写已结束，签名已冻结。现在可以清除或导出。"
      : "书写已结束，本次没有留下笔迹。";
    showToast(message, { duration: 3000 });
    announce(message);
  }
  return true;
}

function rewriteSignature() {
  if (state.writing !== "finished" || !canBeginWriting() || state.calibrationActive) return;
  clearInk();
  startWriting({ showFeedback: false });
  syncUi();
  showToast("旧签名已清除，已重新进入书写模式", { duration: 2800 });
  announce("旧签名已清除，已重新进入书写模式");
}

function clearSignature() {
  if (state.writing !== "finished" || !hasInk() || state.calibrationActive) return;
  clearInk();
  syncUi();
  showToast("签名已清除，书写保持结束状态");
  announce("签名已清除，书写保持结束状态");
}

/* -------------------------------------------------------------------------- */
/* PNG export                                                                 */
/* -------------------------------------------------------------------------- */

function drawExportSignature(ctx) {
  const lineWidth = getInkWidth(EXPORT_WIDTH);
  ctx.strokeStyle = "#17231f";
  ctx.fillStyle = "#17231f";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const stroke of state.strokes) {
    drawInkStroke(ctx, stroke, EXPORT_WIDTH, EXPORT_HEIGHT, lineWidth);
  }
}

function createExportCanvas(background) {
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = EXPORT_WIDTH;
  exportCanvas.height = EXPORT_HEIGHT;
  const ctx = exportCanvas.getContext("2d");
  if (!ctx) return null;

  if (background === "white") {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, EXPORT_WIDTH, EXPORT_HEIGHT);
  } else {
    ctx.clearRect(0, 0, EXPORT_WIDTH, EXPORT_HEIGHT);
  }
  drawExportSignature(ctx);
  return exportCanvas;
}

function getInkBounds() {
  if (!state.strokes.length) return null;
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const stroke of state.strokes) {
    for (const point of stroke) {
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x);
      maxY = Math.max(maxY, point.y);
    }
  }
  return { minX, minY, maxX, maxY };
}

function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    if (typeof canvas.toBlob === "function") {
      canvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error("浏览器未能生成 PNG 文件"));
      }, "image/png");
      return;
    }

    try {
      const dataUrl = canvas.toDataURL("image/png");
      const binary = atob(dataUrl.split(",")[1]);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
      resolve(new Blob([bytes], { type: "image/png" }));
    } catch (error) {
      reject(error);
    }
  });
}

function updateExportPreviewBackground(background) {
  state.exportBackground = background;
  for (const radio of exportRadios) radio.checked = radio.value === background;
  for (const button of previewBgButtons) {
    button.setAttribute("aria-pressed", String(button.dataset.background === background));
  }
  if (els.exportPreviewStage) els.exportPreviewStage.dataset.bg = background;
  if (els.exportDialog && els.exportDialog.open) {
    renderExportPreview(background);
  }
}

function renderExportPreview(background = state.exportBackground) {
  const exportCanvas = createExportCanvas(background);
  if (!exportCanvas || !els.exportPreviewCanvas) return;

  const previewCanvas = els.exportPreviewCanvas;
  const previewWidth = 1200;
  const previewHeight = 500;
  if (previewCanvas.width !== previewWidth || previewCanvas.height !== previewHeight) {
    previewCanvas.width = previewWidth;
    previewCanvas.height = previewHeight;
  }
  const previewCtx = previewCanvas.getContext("2d");
  if (!previewCtx) return;
  previewCtx.clearRect(0, 0, previewWidth, previewHeight);
  previewCtx.drawImage(exportCanvas, 0, 0, previewWidth, previewHeight);
  if (els.exportPreviewStage) els.exportPreviewStage.dataset.bg = background;

  const bounds = getInkBounds();
  let warning = false;
  let message = "边缘完整，已保留签名安全边距";
  if (!bounds) {
    message = "尚无笔迹";
    warning = true;
  } else if (
    bounds.minX <= SAFE_AREA.left - 0.015 ||
    bounds.maxX >= SAFE_AREA.left + SAFE_AREA.width + 0.015 ||
    bounds.minY <= SAFE_AREA.top - 0.015 ||
    bounds.maxY >= SAFE_AREA.top + SAFE_AREA.height + 0.015
  ) {
    message = "部分笔迹接近画布边缘，导出仍为完整画布";
    warning = true;
  }
  if (els.exportEdgeStatus) {
    els.exportEdgeStatus.textContent = message;
    els.exportEdgeStatus.dataset.warning = String(warning);
  }
}

function openExportDialog() {
  if (state.writing !== "finished" || !hasInk() || state.exporting) return;
  const background = exportRadios.find((radio) => radio.checked)?.value || "white";
  state.exportBackground = background;
  updateExportPreviewBackground(background);
  renderExportPreview(background);
  if (typeof els.exportDialog.showModal === "function") {
    if (!els.exportDialog.open) els.exportDialog.showModal();
  } else {
    els.exportDialog.setAttribute("open", "");
  }
  announce("已打开签名导出预览");
}

function downloadExportPreview() {
  if (state.writing !== "finished" || !hasInk() || state.exporting) return;
  const background = state.exportBackground || "white";
  const exportCanvas = createExportCanvas(background);
  if (!exportCanvas) {
    showToast("当前浏览器无法创建导出画布，请更换浏览器后重试", { error: true, duration: 4200 });
    return;
  }

  state.exporting = true;
  if (els.downloadExportBtn) els.downloadExportBtn.disabled = true;
  if (els.downloadExportBtnText) els.downloadExportBtnText.textContent = "正在生成…";
  syncUi();

  canvasToBlob(exportCanvas)
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const now = new Date();
      const stamp = [
        now.getFullYear(),
        String(now.getMonth() + 1).padStart(2, "0"),
        String(now.getDate()).padStart(2, "0"),
        "-",
        String(now.getHours()).padStart(2, "0"),
        String(now.getMinutes()).padStart(2, "0"),
        String(now.getSeconds()).padStart(2, "0"),
      ].join("");
      link.href = url;
      link.download = `kongshu-signature-${stamp}.png`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1500);

      const message = `签名已导出（${background === "white" ? "白色背景" : "透明背景"}）`;
      showToast(message, { duration: 3200 });
      announce(message);
      if (els.exportEdgeStatus) {
        els.exportEdgeStatus.textContent = "PNG 已下载，可切换背景后继续下载";
        els.exportEdgeStatus.dataset.warning = "false";
      }
    })
    .catch((error) => {
      console.error("导出签名失败", error);
      showToast("导出失败，请重试或更换浏览器", { error: true, duration: 4200 });
      if (els.exportEdgeStatus) {
        els.exportEdgeStatus.textContent = "导出失败，请重试或更换浏览器";
        els.exportEdgeStatus.dataset.warning = "true";
      }
      announce("导出签名失败");
    })
    .finally(() => {
      state.exporting = false;
      if (els.downloadExportBtn) els.downloadExportBtn.disabled = false;
      if (els.downloadExportBtnText) els.downloadExportBtnText.textContent = "下载 PNG";
      syncUi();
    });
}

function undoLastStroke() {
  if (state.writing !== "finished" || !state.strokes.length) return;
  state.strokes.pop();
  state.currentStroke = null;
  renderSignature();
  syncUi();
  showToast("已撤销最后一笔");
  announce("已撤销最后一笔");
}

/* -------------------------------------------------------------------------- */
/* Overlay drawing                                                            */
/* -------------------------------------------------------------------------- */

function getProjection(width, height) {
  const videoWidth = els.cameraVideo.videoWidth || 1280;
  const videoHeight = els.cameraVideo.videoHeight || 720;
  const scale = Math.max(width / videoWidth, height / videoHeight);
  const drawWidth = videoWidth * scale;
  const drawHeight = videoHeight * scale;
  return {
    dx: (width - drawWidth) / 2,
    dy: (height - drawHeight) / 2,
    drawWidth,
    drawHeight,
  };
}

function perceivedToScreen(x, y, projection) {
  return {
    x: projection.dx + x * projection.drawWidth,
    y: projection.dy + y * projection.drawHeight,
  };
}

function rawToScreen(point, projection) {
  const x = settings.mirror ? 1 - point.x : point.x;
  return {
    x: projection.dx + x * projection.drawWidth,
    y: projection.dy + point.y * projection.drawHeight,
  };
}

function drawOverlay() {
  if (!overlayCtx) return;
  if (!state.overlayW || !state.overlayH) return;

  overlayCtx.clearRect(0, 0, state.overlayW, state.overlayH);
  if (state.camera !== "live") return;

  const projection = getProjection(state.overlayW, state.overlayH);
  if (!state.calibrationActive) drawRoi(projection);

  const landmarks = state.latestLandmarks;
  if (!landmarks) return;

  const scale = clamp(state.overlayW / 680, 0.72, 1.55);
  const points = landmarks.map((point) => rawToScreen(point, projection));

  overlayCtx.save();
  overlayCtx.lineCap = "round";
  overlayCtx.lineJoin = "round";

  const drawConnections = (highlight) => {
    overlayCtx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      const isIndex = INDEX_CHAIN.has(`${a}-${b}`);
      if (highlight && !isIndex) continue;
      overlayCtx.moveTo(points[a].x, points[a].y);
      overlayCtx.lineTo(points[b].x, points[b].y);
    }
    overlayCtx.stroke();
  };

  overlayCtx.strokeStyle = "rgba(255, 248, 234, 0.22)";
  overlayCtx.lineWidth = 4.2 * scale;
  drawConnections(false);
  overlayCtx.strokeStyle = "rgba(255, 248, 234, 0.64)";
  overlayCtx.lineWidth = 1.35 * scale;
  drawConnections(false);

  overlayCtx.strokeStyle = "rgba(235, 195, 111, 0.98)";
  overlayCtx.lineWidth = 2.1 * scale;
  drawConnections(true);

  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const isTip = index === 4 || index === 8;
    const isIndexTip = index === 8;
    const radius = (isTip ? 3.7 : 2.25) * scale;

    overlayCtx.beginPath();
    overlayCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    overlayCtx.fillStyle = isIndexTip
      ? "#e17b5a"
      : isTip
        ? "rgba(255, 248, 234, 0.85)"
        : "rgba(43, 33, 23, 0.8)";
    overlayCtx.fill();
    overlayCtx.lineWidth = 1 * scale;
    overlayCtx.strokeStyle = isIndexTip
      ? "rgba(255, 246, 229, 0.92)"
      : "rgba(255, 248, 234, 0.7)";
    overlayCtx.stroke();
  }

  if (state.pinching) {
    const thumb = points[4];
    const index = points[8];
    overlayCtx.beginPath();
    overlayCtx.setLineDash([3 * scale, 3.6 * scale]);
    overlayCtx.moveTo(thumb.x, thumb.y);
    overlayCtx.lineTo(index.x, index.y);
    overlayCtx.strokeStyle = "rgba(235, 195, 111, 0.82)";
    overlayCtx.lineWidth = 1 * scale;
    overlayCtx.stroke();
    overlayCtx.setLineDash([]);
  }

  const cursor = state.latestCursor
    ? perceivedToScreen(state.latestCursor.x, state.latestCursor.y, projection)
    : rawToScreen(landmarks[8], projection);
  const cursorRadius = 12 * scale;

  if (state.pinching) {
    const glow = overlayCtx.createRadialGradient(
      cursor.x,
      cursor.y,
      0,
      cursor.x,
      cursor.y,
      28 * scale,
    );
    glow.addColorStop(0, "rgba(225, 123, 90, 0.3)");
    glow.addColorStop(1, "rgba(225, 123, 90, 0)");
    overlayCtx.fillStyle = glow;
    overlayCtx.beginPath();
    overlayCtx.arc(cursor.x, cursor.y, 28 * scale, 0, Math.PI * 2);
    overlayCtx.fill();
  }

  overlayCtx.beginPath();
  overlayCtx.arc(cursor.x, cursor.y, cursorRadius, 0, Math.PI * 2);
  overlayCtx.strokeStyle = state.pinching ? "#e17b5a" : "rgba(255, 244, 227, 0.9)";
  overlayCtx.lineWidth = (state.pinching ? 2 : 1.35) * scale;
  overlayCtx.stroke();

  overlayCtx.beginPath();
  overlayCtx.arc(cursor.x, cursor.y, state.pinching ? 3 * scale : 2 * scale, 0, Math.PI * 2);
  overlayCtx.fillStyle = state.pinching ? "#fff5e4" : "rgba(43, 33, 23, 0.86)";
  overlayCtx.fill();

  overlayCtx.beginPath();
  overlayCtx.moveTo(cursor.x - (cursorRadius + 5) * scale, cursor.y);
  overlayCtx.lineTo(cursor.x - (cursorRadius + 1) * scale, cursor.y);
  overlayCtx.moveTo(cursor.x + (cursorRadius + 1) * scale, cursor.y);
  overlayCtx.lineTo(cursor.x + (cursorRadius + 5) * scale, cursor.y);
  overlayCtx.moveTo(cursor.x, cursor.y - (cursorRadius + 5) * scale);
  overlayCtx.lineTo(cursor.x, cursor.y - (cursorRadius + 1) * scale);
  overlayCtx.moveTo(cursor.x, cursor.y + (cursorRadius + 1) * scale);
  overlayCtx.lineTo(cursor.x, cursor.y + (cursorRadius + 5) * scale);
  overlayCtx.strokeStyle = "rgba(255, 244, 227, 0.55)";
  overlayCtx.lineWidth = 0.8 * scale;
  overlayCtx.stroke();

  if (state.calibrationActive) {
    drawCalibrationMarks(projection, scale);
  }

  overlayCtx.restore();
}

function drawCalibrationMarks(projection, scale) {
  const points = state.calibrationPoints;
  if (!points.length) return;

  overlayCtx.save();
  overlayCtx.lineCap = "round";
  overlayCtx.lineJoin = "round";

  if (points.length >= 2) {
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    const topLeft = perceivedToScreen(Math.min(...xs), Math.min(...ys), projection);
    const bottomRight = perceivedToScreen(Math.max(...xs), Math.max(...ys), projection);
    overlayCtx.fillStyle = "rgba(255, 248, 234, 0.035)";
    overlayCtx.fillRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
    overlayCtx.strokeStyle = "rgba(235, 195, 111, 0.68)";
    overlayCtx.lineWidth = 1;
    overlayCtx.setLineDash([4 * scale, 5 * scale]);
    overlayCtx.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
    overlayCtx.setLineDash([]);
  }

  points.forEach((point, index) => {
    const screen = perceivedToScreen(point.x, point.y, projection);
    overlayCtx.beginPath();
    overlayCtx.arc(screen.x, screen.y, 8 * scale, 0, Math.PI * 2);
    overlayCtx.fillStyle = "rgba(178, 87, 63, 0.9)";
    overlayCtx.fill();
    overlayCtx.strokeStyle = "rgba(255, 248, 234, 0.9)";
    overlayCtx.lineWidth = 1;
    overlayCtx.stroke();
    overlayCtx.fillStyle = "#fffaf0";
    overlayCtx.font = `500 ${9 * scale}px "Avenir Next", "PingFang SC", sans-serif`;
    overlayCtx.textAlign = "center";
    overlayCtx.textBaseline = "middle";
    overlayCtx.fillText(String(index + 1), screen.x, screen.y + 0.5);
  });

  if (points.length >= 2) {
    const last = points[points.length - 1];
    const screen = perceivedToScreen(last.x, last.y, projection);
    const label = points.length < 4 ? `已记录 ${points.length} / 4` : "范围完成";
    overlayCtx.font = `500 ${9 * scale}px "Avenir Next", "PingFang SC", sans-serif`;
    const textWidth = overlayCtx.measureText(label).width;
    const pad = 7;
    const boxW = textWidth + pad * 2;
    const boxH = 18;
    const boxX = clamp(screen.x - boxW / 2, 8, state.overlayW - boxW - 8);
    const boxY = clamp(screen.y + 14 * scale, 8, state.overlayH - boxH - 8);
    roundedRectPath(overlayCtx, boxX, boxY, boxW, boxH, 4);
    overlayCtx.fillStyle = "rgba(48, 41, 31, 0.7)";
    overlayCtx.fill();
    overlayCtx.fillStyle = "rgba(255, 248, 234, 0.86)";
    overlayCtx.textAlign = "left";
    overlayCtx.textBaseline = "middle";
    overlayCtx.fillText(label, boxX + pad, boxY + boxH / 2);
  }
  overlayCtx.restore();
}

function drawRoi(projection) {
  const scale = clamp(state.overlayW / 680, 0.72, 1.55);
  const roi = getActiveRoi();
  const topLeft = perceivedToScreen(roi.minX, roi.minY, projection);
  const bottomRight = perceivedToScreen(roi.maxX, roi.maxY, projection);
  const width = bottomRight.x - topLeft.x;
  const height = bottomRight.y - topLeft.y;

  overlayCtx.save();
  overlayCtx.fillStyle = "rgba(255, 248, 234, 0.02)";
  overlayCtx.fillRect(topLeft.x, topLeft.y, width, height);
  overlayCtx.strokeStyle = "rgba(255, 248, 234, 0.34)";
  overlayCtx.lineWidth = 1;
  overlayCtx.setLineDash([5 * scale, 6 * scale]);
  overlayCtx.strokeRect(topLeft.x, topLeft.y, width, height);
  overlayCtx.setLineDash([]);

  const bracket = 12 * scale;
  overlayCtx.strokeStyle = "rgba(213, 174, 92, 0.86)";
  overlayCtx.lineWidth = 1.2;
  const drawBracket = (x, y, dx, dy) => {
    overlayCtx.beginPath();
    overlayCtx.moveTo(x + dx * bracket, y);
    overlayCtx.lineTo(x, y);
    overlayCtx.lineTo(x, y + dy * bracket);
    overlayCtx.stroke();
  };
  drawBracket(topLeft.x, topLeft.y, 1, 1);
  drawBracket(bottomRight.x, topLeft.y, -1, 1);
  drawBracket(topLeft.x, bottomRight.y, 1, -1);
  drawBracket(bottomRight.x, bottomRight.y, -1, -1);

  const fontSize = clamp(10 * scale, 9, 12);
  overlayCtx.font = `500 ${fontSize}px "PingFang SC", "Microsoft YaHei", sans-serif`;
  const label = state.calibrationRange ? "已校准范围" : "书写范围";
  const textWidth = overlayCtx.measureText(label).width;
  const padX = 7;
  const labelHeight = 20;
  const labelX = bottomRight.x - textWidth - padX * 2 - 9;
  const labelY = Math.max(topLeft.y + 9, 8);
  roundedRectPath(overlayCtx, labelX, labelY, textWidth + padX * 2, labelHeight, 5);
  overlayCtx.fillStyle = "rgba(48, 41, 31, 0.7)";
  overlayCtx.fill();
  overlayCtx.fillStyle = "rgba(255, 248, 234, 0.82)";
  overlayCtx.fillText(label, labelX + padX, labelY + labelHeight / 2 + fontSize * 0.35);
  overlayCtx.restore();
}

function roundedRectPath(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

/* -------------------------------------------------------------------------- */
/* Event wiring                                                               */
/* -------------------------------------------------------------------------- */

function bindEvents() {
  els.cameraStartBtn.addEventListener("click", startCamera);
  els.cameraStopBtn.addEventListener("click", () => stopCamera());

  els.toggleWritingBtn.addEventListener("click", () => {
    if (state.writing === "writing") finishWriting();
    else if (state.writing === "idle") startWriting();
  });
  els.rewriteBtn.addEventListener("click", rewriteSignature);
  els.undoBtn.addEventListener("click", undoLastStroke);
  els.clearBtn.addEventListener("click", clearSignature);
  els.exportBtn.addEventListener("click", openExportDialog);

  els.calibrateBtn.addEventListener("click", startCalibration);
  els.settingsCalibrateBtn.addEventListener("click", startCalibration);
  els.calibrationCaptureBtn.addEventListener("click", captureCalibrationPoint);
  els.calibrationSkipBtn.addEventListener("click", skipCalibration);
  els.calibrationDefaultBtn.addEventListener("click", resetCalibrationRange);
  els.resetCalibrationBtn.addEventListener("click", resetCalibrationRange);
  els.previewBackgroundSwitch.addEventListener("click", (event) => {
    const button = event.target.closest(".preview-bg-option");
    if (!button) return;
    updateExportPreviewBackground(button.dataset.background);
  });
  els.downloadExportBtn.addEventListener("click", downloadExportPreview);

  els.alertActionBtn.addEventListener("click", () => {
    const action = state.alertAction;
    if (typeof action === "function") action();
  });

  els.mirrorToggle.addEventListener("change", () => {
    if (state.calibrationActive) return;
    settings.mirror = els.mirrorToggle.checked;
    els.cameraVideo.classList.toggle("mirrored", settings.mirror);
    mirrorCalibrationRange();
    state.lastWristPerceived = null;
    resetFingerFilters();
    updateDirectionLabels();
    persistSettings();
    handleLiveMappingChange("画面方向已切换，请松开手指后重新捏合。");
    drawOverlay();
  });

  els.directionMode.addEventListener("change", () => {
    settings.directionMode = els.directionMode.value === "invert" ? "invert" : "follow";
    resetFingerFilters();
    updateDirectionLabels();
    persistSettings();
    handleLiveMappingChange("左右映射已调整，请松开手指后重新捏合。");
  });

  els.pinchThreshold.addEventListener("input", () => {
    settings.threshold = clamp(Number(els.pinchThreshold.value), 0.2, 0.6);
    els.pinchThresholdValue.textContent = pinchThresholdLabel(settings.threshold);
    els.pinchThreshold.setAttribute("aria-valuetext", pinchThresholdLabel(settings.threshold));
    updatePinchMark();
    persistSettings();
  });

  els.releaseDelay.addEventListener("input", () => {
    settings.releaseDelay = clamp(Math.round(Number(els.releaseDelay.value) / 20) * 20, 60, 300);
    els.releaseDelayValue.textContent = `${releaseDelayLabel(settings.releaseDelay)} · ${settings.releaseDelay}ms`;
    els.releaseDelay.setAttribute(
      "aria-valuetext",
      `${releaseDelayLabel(settings.releaseDelay)}，${settings.releaseDelay} 毫秒`,
    );
    persistSettings();
  });

  els.mapGain.addEventListener("input", () => {
    settings.mapGain = clamp(Number(els.mapGain.value) / 100, 0.7, 1.3);
    els.mapGainValue.textContent = `${Math.round(settings.mapGain * 100)}%`;
    if (state.writing === "writing") {
      endStroke();
      state.blockUntilRelease = state.pinching;
      syncUi();
      showToast("笔迹跟随幅度已调整，请松开手指后重新落笔");
    }
    persistSettings();
  });

  els.penWidth.addEventListener("input", () => {
    settings.penWidth = clamp(Math.round(Number(els.penWidth.value)), 1, 6);
    els.penWidthValue.textContent = penWidthLabel(settings.penWidth);
    renderSignature();
    persistSettings();
  });

  els.smoothing.addEventListener("input", () => {
    settings.smoothing = clamp(Math.round(Number(els.smoothing.value)), 0, 100);
    els.smoothingValue.textContent = smoothingLabel(settings.smoothing);
    updateFilterSmoothing();
    persistSettings();
  });

  els.resetSettingsBtn.addEventListener("click", () => {
    settings = { ...DEFAULT_SETTINGS };
    resetFingerFilters();
    syncSettingControls();
    renderSignature();
    persistSettings();
    handleLiveMappingChange("已恢复默认设置。请松开手指后重新捏合。");
    showToast("已恢复默认手势与笔迹设置");
  });

  els.helpButton.addEventListener("click", () => {
    if (typeof els.helpDialog.showModal === "function") els.helpDialog.showModal();
    else els.helpDialog.setAttribute("open", "");
  });

  els.helpDialog.addEventListener("click", (event) => {
    if (event.target === els.helpDialog) els.helpDialog.close();
  });

  els.helpDialog.addEventListener("close", () => {
    els.helpButton.focus();
  });

  for (const radio of exportRadios) {
    radio.addEventListener("change", () => {
      if (radio.checked) {
        updateExportPreviewBackground(radio.value);
        showToast(radio.value === "white" ? "导出时将使用白色背景" : "导出时将使用透明背景", {
          duration: 1900,
        });
      }
    });
  }

  document.addEventListener("keydown", (event) => {
    if (
      state.calibrationActive &&
      event.code === "Space" &&
      !event.repeat &&
      !document.querySelector("dialog[open]")
    ) {
      event.preventDefault();
      captureCalibrationPoint();
    }
  });

  els.exportDialog.addEventListener("click", (event) => {
    if (event.target === els.exportDialog) els.exportDialog.close();
  });

  els.exportDialog.addEventListener("close", () => {
    if (!els.exportBtn.disabled) els.exportBtn.focus();
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (state.writing === "writing") {
        endStroke();
        state.blockUntilRelease = true;
        state.handDetected = false;
        state.pinching = false;
        state.pinchRatio = null;
        state.pinchRatioSmoothed = null;
        state.pinchFrames = 0;
        state.releaseFrames = 0;
        state.releaseSince = null;
        state.drawBlockedUntil = Number.POSITIVE_INFINITY;
        state.stabilityReason = "reacquire";
        state.latestLandmarks = null;
        state.latestCursor = null;
        state.latestCanvasPoint = null;
        state.latestRawCanvasPoint = null;
        drawOverlay();
        syncUi();
      }
    } else {
      state.lastVideoTime = -1;
      resetFingerFilters();
    }
  });

  window.addEventListener("pagehide", () => {
    stopTracking();
    releaseStream();
  });

  window.addEventListener("beforeunload", () => {
    stopTracking();
    releaseStream();
  });

  window.addEventListener("resize", () => {
    fitOverlayCanvas();
    fitSignatureCanvas();
  });

  if (typeof ResizeObserver !== "undefined") {
    const overlayObserver = new ResizeObserver(() => fitOverlayCanvas());
    overlayObserver.observe(els.cameraFrame);
    const signatureObserver = new ResizeObserver(() => fitSignatureCanvas());
    signatureObserver.observe(els.signatureShell);
  }
}

function handleLiveMappingChange(message) {
  if (state.writing === "writing") {
    endStroke();
    state.blockUntilRelease = state.pinching;
    syncUi();
    if (state.pinching) showToast(message, { duration: 3200 });
  } else {
    syncUi();
  }
  drawOverlay();
}

/* -------------------------------------------------------------------------- */
/* Init                                                                       */
/* -------------------------------------------------------------------------- */

function init() {
  loadSettings();
  loadCalibration();
  state.exportBackground = exportRadios.find((radio) => radio.checked)?.value || "white";
  syncSettingControls();
  resetFingerFilters();
  bindEvents();
  syncUi();
  updateCalibrationUi();
  updatePinchMark();

  requestAnimationFrame(() => {
    fitOverlayCanvas();
    fitSignatureCanvas();
  });
}

init();
