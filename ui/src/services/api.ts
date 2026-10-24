/**
 * API abstraction layer for the HemaVision AI frontend.
 *
 * This project keeps the original Python backend and dataset-driven processing.
 * The frontend therefore connects to the real FastAPI endpoints instead of the
 * mock dataset, while still caching recent results in local browser storage so
 * the dashboard and result pages remain usable without adding new server routes.
 */
import type {
  Analysis,
  AnalysisSummary,
  AnalyzeRequest,
  DashboardStats,
  ModelStatus,
  Report,
  Detection,
} from "@/types";

export const API_BASE_URL =
  (import.meta.env["VITE_API_URL"] as string | undefined) ||
  "http://127.0.0.1:8000";

export const USE_MOCK_DATA = import.meta.env["VITE_USE_MOCK_DATA"] === "true";

const STORAGE_KEY = "hemavision-analyses";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function friendlyMessage(status: number) {
  if (status === 0) return "Unable to connect to the AI analysis server.";
  if (status === 404) return "The requested resource could not be found.";
  if (status === 413) return "The uploaded image is too large.";
  if (status >= 500) return "The AI analysis server encountered a problem.";
  return "The request could not be completed.";
}

function parseNumber(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function readLocalAnalyses(): Analysis[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Analysis[]) : [];
  } catch {
    return [];
  }
}

function writeLocalAnalyses(items: Analysis[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
}

// The YOLO backend returns absolute pixel coords in 640×640 preprocessed space.
// We normalise to 0–1 fractions here so ImageViewer can render with CSS percentages.
const YOLO_IMG_SIZE = 640;

function normaliseDetection(item: any): Detection {
  const bbox = Array.isArray(item?.bbox) ? item.bbox : [0, 0, 0, 0];
  const [x1, y1, x2, y2] = bbox.map((v: unknown) => parseNumber(v, 0));

  // Determine if coords are already normalised (all <= 1) or pixel-space
  const isAlreadyNorm = x2 <= 1.5 && y2 <= 1.5;
  const scale = isAlreadyNorm ? 1 : YOLO_IMG_SIZE;

  const nx1 = x1 / scale;
  const ny1 = y1 / scale;
  const nx2 = x2 / scale;
  const ny2 = y2 / scale;

  // Map backend class name (lowercase: "rbc", "wbc", "platelet") to display name
  const rawCls = String(item?.cls ?? item?.class ?? "rbc").toLowerCase();
  const cellType: "RBC" | "WBC" | "Platelet" =
    rawCls.includes("wbc") ? "WBC" :
    rawCls.includes("plat") ? "Platelet" : "RBC";

  return {
    id: `${cellType}-${Math.random().toString(36).slice(2, 9)}`,
    cellType,
    confidence: parseNumber(item?.confidence, 0),
    bbox: {
      x:      Math.max(0, Math.min(nx1, 1)),
      y:      Math.max(0, Math.min(ny1, 1)),
      width:  Math.max(0, Math.min(nx2 - nx1, 1)),
      height: Math.max(0, Math.min(ny2 - ny1, 1)),
    },
  };
}

function riskLevelFromOverall(overall?: string): "low" | "moderate" | "high" {
  if (!overall) return "low";
  const value = overall.toLowerCase();
  if (value.includes("abnormal") || value.includes("anemia") || value.includes("high") || value.includes("low")) {
    return value.includes("severe") ? "high" : "moderate";
  }
  return "low";
}

function toUiAnalysis(data: any, file?: File): Analysis {
  const raw = data?.raw_counts ?? {};
  const calibrated = data?.calibrated_counts ?? {};
  const diff = data?.wbc_differential ?? {};
  const morphology = data?.morphology ?? {};
  const anemiaRisk = data?.anemia_risk ?? {};
  const cellFlags = Array.isArray(data?.cell_flags) ? data.cell_flags.map((flag: any) => ({
    cellType: String(flag?.cell_type ?? "Unknown"),
    label: String(flag?.label ?? "Unknown"),
    value: parseNumber(flag?.value, 0),
    unit: String(flag?.unit ?? ""),
    severity: String(flag?.severity ?? "Low"),
    condition: String(flag?.condition ?? ""),
    reference: String(flag?.reference ?? ""),
  })) : [];
  const diffFlags = Array.isArray(data?.diff_flags) ? data.diff_flags.map((flag: any) => ({
    cellType: String(flag?.cell_type ?? "Unknown"),
    label: String(flag?.label ?? "Unknown"),
    value: parseNumber(flag?.value, 0),
    unit: String(flag?.unit ?? ""),
    severity: String(flag?.severity ?? "Low"),
    condition: String(flag?.condition ?? ""),
    reference: String(flag?.reference ?? ""),
  })) : [];
  const detections = Array.isArray(data?.detections) ? data.detections.map(normaliseDetection) : [];
  const overall = String(data?.overall ?? "Normal");
  const summaryItems = Array.isArray(data?.summary) ? data.summary : [overall];
  const summaryText = summaryItems.join(" ");
  const risk = riskLevelFromOverall(overall) || (anemiaRisk?.level as "low" | "moderate" | "high") || "low";
  const total = Math.max(1, detections.length || Number(raw.rbc ?? 0) + Number(raw.wbc ?? 0) + Number(raw.platelet ?? 0));
  const cellKeys = [
    { key: "rbc", label: "RBC", unit: "M/µL", ref: "4.5 – 5.9 M/µL" },
    { key: "wbc", label: "WBC", unit: "K/µL", ref: "4.0 – 11.0 K/µL" },
    { key: "platelet", label: "Platelet", unit: "K/µL", ref: "150 – 450 K/µL" },
  ] as const;

  const summary = cellKeys.map(({ key, label, unit, ref }) => {
    const value = parseNumber(calibrated?.[key], 0);
    const fixedValue = key === "rbc" ? (value / 1000000).toFixed(2) : (value / 1000).toFixed(1);
    const status: "normal" | "low" | "high" | "unknown" = key === "rbc" ? (value < 4.5 ? "low" : value > 5.9 ? "high" : "normal") : key === "wbc" ? (value < 4 ? "low" : value > 11 ? "high" : "normal") : value < 150 ? "low" : value > 450 ? "high" : "normal";
    return {
      cellType: label,
      value: fixedValue,
      unit,
      referenceRange: ref,
      status,
      confidence: parseNumber(data?.avg_confidence, 0.9),
    };
  });

  const breakdown = cellKeys.map(({ key, label }) => ({
    cellType: label,
    count: parseNumber(calibrated?.[key], 0),
    averageConfidence: parseNumber(data?.avg_confidence, 0.9),
    percentage: Number(((parseNumber(calibrated?.[key], 0) / total) * 100 || 0).toFixed(1)),
  }));

  const morphologyMetrics = {
    averageCellArea: parseNumber(morphology?.average_cell_area ?? morphology?.averageCellArea, 0),
    circularity: parseNumber(morphology?.circularity, 0),
    aspectRatio: parseNumber(morphology?.aspect_ratio ?? morphology?.aspectRatio, 0),
    meanIntensity: parseNumber(morphology?.mean_intensity ?? morphology?.meanIntensity, 0),
    shapeVariation: (morphology?.shape_variation ?? morphology?.shapeVariation ?? "Moderate") as "Low" | "Moderate" | "High",
    areaDistribution: Array.isArray(morphology?.area_distribution ?? morphology?.areaDistribution)
      ? (morphology?.area_distribution ?? morphology?.areaDistribution).map((item: any) => ({
          bin: String(item?.bin ?? "0-10"),
          count: parseNumber(item?.count, 0),
        }))
      : [],
    profile: Array.isArray(morphology?.profile) ? morphology.profile.map((item: any) => ({
      metric: String(item?.metric ?? "Metric"),
      value: parseNumber(item?.value, 0),
    })) : [],
  };

  const anemiaMetrics = {
    score: parseNumber(anemiaRisk?.score, overall.toLowerCase().includes("normal") ? 18 : 74),
    level: (anemiaRisk?.level ?? risk ?? "low") as "low" | "moderate" | "high",
    explanation: String(anemiaRisk?.explanation ?? summaryText),
    contributingFactors: Array.isArray(anemiaRisk?.contributing_factors ?? anemiaRisk?.contributingFactors)
      ? (anemiaRisk?.contributing_factors ?? anemiaRisk?.contributingFactors).map((item: any) => ({
          factor: String(item?.factor ?? "RBC density"),
          weight: parseNumber(item?.weight, 0),
        }))
      : [
          { factor: "RBC density", weight: 0.42 },
          { factor: "Cell morphology", weight: 0.33 },
          { factor: "WBC differential", weight: 0.25 },
        ],
  };

    const id = String(data?.analysis_id ?? `ANL-${Date.now()}`);
  const imageUrl = typeof file !== "undefined" && file ? URL.createObjectURL(file) : "";

  let anemiaImage: any = undefined;
  if (data?.anemia_image) {
    anemiaImage = {
      prediction: String(data.anemia_image.prediction ?? "Undetermined"),
      isAnemic: Boolean(data.anemia_image.is_anemic),
      probability: parseNumber(data.anemia_image.probability, 0),
      confidence: parseNumber(data.anemia_image.confidence, 0),
      modelName: String(data.anemia_image.model_name ?? "MobileNetV3-Small"),
      modelVersion: String(data.anemia_image.model_version ?? "1.0.0"),
      status: String(data.anemia_image.status ?? "success"),
      message: String(data.anemia_image.message ?? ""),
    };
  }

  let anemiaFusion: any = undefined;
  if (data?.anemia_fusion) {
    anemiaFusion = {
      fusionPrediction: String(data.anemia_fusion.fusion_prediction ?? "Undetermined"),
      isAnemic: Boolean(data.anemia_fusion.is_anemic),
      fusionProbability: parseNumber(data.anemia_fusion.fusion_probability, 0),
      fusionConfidence: parseNumber(data.anemia_fusion.fusion_confidence, 0),
      fusionStrategy: String(data.anemia_fusion.fusion_strategy ?? "none"),
      imageProbability: parseNumber(data.anemia_fusion.image_probability, 0),
      cbcProbability: parseNumber(data.anemia_fusion.cbc_probability, 0),
      modelVersion: String(data.anemia_fusion.model_version ?? "1.0.0"),
      status: String(data.anemia_fusion.status ?? "success"),
      message: String(data.anemia_fusion.message ?? ""),
      fusionMode: String(data.anemia_fusion.fusion_mode ?? (data.anemia_fusion.is_experimental ? "experimental_gmu" : "default")),
      latentGateMean: (data.anemia_fusion.latent_gate_mean !== undefined && data.anemia_fusion.latent_gate_mean !== null)
        ? parseNumber(data.anemia_fusion.latent_gate_mean, 0)
        : undefined,
      isExperimental: Boolean(data.anemia_fusion.is_experimental),
      limitationNotice: data.anemia_fusion.limitation_notice ? String(data.anemia_fusion.limitation_notice) : undefined,
    };
  }

  let anemiaMl: any = undefined;
  if (data?.anemia_ml) {
    const rawCbc = data.anemia_ml.cbc_inputs || data.anemia_ml.cbcInputs || data.cbc_inputs_json;
    const cbcInputs = rawCbc ? {
      Gender: rawCbc.Gender ?? rawCbc.gender,
      Hemoglobin: rawCbc.Hemoglobin ?? rawCbc.hb ?? rawCbc.hemoglobin,
      MCV: rawCbc.MCV ?? rawCbc.mcv,
      MCH: rawCbc.MCH ?? rawCbc.mch,
      MCHC: rawCbc.MCHC ?? rawCbc.mchc,
    } : undefined;

    anemiaMl = {
      prediction: String(data.anemia_ml.prediction ?? "Undetermined"),
      isAnemic: Boolean(data.anemia_ml.is_anemic),
      probability: parseNumber(data.anemia_ml.probability, 0),
      confidence: parseNumber(data.anemia_ml.confidence, 0),
      modelName: String(data.anemia_ml.model_name ?? "Random Forest"),
      modelVersion: String(data.anemia_ml.model_version ?? "1.0.0"),
      status: String(data.anemia_ml.status ?? "success"),
      message: String(data.anemia_ml.message ?? ""),
      cbcInputs,
    };
  }

  const topRawCbc = data?.cbc_inputs || data?.cbcInputs || data?.cbc_inputs_json || data?.anemia_ml?.cbc_inputs || data?.anemia_ml?.cbcInputs;
  const topCbcInputs = topRawCbc ? {
    Gender: topRawCbc.Gender ?? topRawCbc.gender,
    Hemoglobin: topRawCbc.Hemoglobin ?? topRawCbc.hb ?? topRawCbc.hemoglobin,
    MCV: topRawCbc.MCV ?? topRawCbc.mcv,
    MCH: topRawCbc.MCH ?? topRawCbc.mch,
    MCHC: topRawCbc.MCHC ?? topRawCbc.mchc,
  } : undefined;

  return {
    id,
    createdAt: new Date().toISOString(),
    imageUrl,
    processedImageUrl: imageUrl,
    imageName: file?.name ?? "blood-smear.jpg",
    patientId: undefined,
    sampleId: undefined,
    notes: undefined,
    processingTime: 0,
    status: "completed",
    confidence: parseNumber(data?.avg_confidence, 0),
    detections,
    cellStatistics: {
      totalDetected: total,
      summary,
      breakdown,
    },
    morphology: morphologyMetrics,
    anemiaRisk: anemiaMetrics,
    explainability: {
      note: summaryText,
      confidenceByClass: [
        { cellType: "RBC", confidence: parseNumber(data?.avg_confidence, 0.9) },
        { cellType: "WBC", confidence: parseNumber(data?.avg_confidence, 0.86) },
        { cellType: "Platelet", confidence: parseNumber(data?.avg_confidence, 0.81) },
      ],
      featureSummary: [
        { feature: "Detection confidence", value: `${parseNumber(data?.avg_confidence, 0)}%` },
        { feature: "Overall outcome", value: overall },
        { feature: "WBC differential", value: Object.keys(diff).length ? `${Object.entries(diff).map(([k, v]) => `${k}: ${Number(v).toFixed(1)}%`).join(", ")}` : "Not available" },
      ],
    },
    rawCounts: {
      rbc: parseNumber(raw.rbc, 0),
      wbc: parseNumber(raw.wbc, 0),
      platelet: parseNumber(raw.platelet, 0),
    },
    calibratedCounts: {
      rbc: parseNumber(calibrated.rbc, 0),
      wbc: parseNumber(calibrated.wbc, 0),
      platelet: parseNumber(calibrated.platelet, 0),
    },
    wbcDifferential: diff,
    cellFlags,
    diffFlags,
    overall,
    summary: summaryItems,
    reportAvailable: Boolean(data?.report_available),
    reportUrl: data?.report_url ?? undefined,
    isEstimatedDiff: Boolean(data?.is_estimated_diff),
    anemiaMl,
    anemiaImage,
    anemiaFusion,
    cbcInputs: topCbcInputs,
  };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      ...(init?.body instanceof FormData ? {} : { headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) } }),
    });
  } catch {
    throw new ApiError(friendlyMessage(0), 0);
  }

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    const message = detail ? JSON.parse(detail).detail ?? detail : friendlyMessage(response.status);
    throw new ApiError(message, response.status);
  }

  return (await response.json()) as T;
}

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const api = {
  async health(): Promise<ModelStatus> {
    if (USE_MOCK_DATA) {
      await delay(300);
      return {
        detectionModel: { name: "YOLOv8", version: "blood-model", loaded: true },
        classificationModel: { name: "ResNet50", version: "wbc-model", loaded: true },
        inferenceEngine: { name: "PyTorch / OpenCV", online: true },
        apiOnline: true,
        frontendVersion: "1.0.0",
      };
    }

    try {
      const res = await request<{ status?: string; version?: string; message?: string }>("/health");
      return {
        detectionModel: { name: "YOLOv8", version: "blood-model", loaded: true },
        classificationModel: { name: "ResNet50", version: "wbc-model", loaded: true },
        inferenceEngine: { name: "PyTorch / OpenCV", online: true },
        apiOnline: res?.status === "ok" || ((res?.message?.length ?? 0) > 0),
        frontendVersion: res?.version ?? "1.0.0",
      };
    } catch {
      return {
        detectionModel: { name: "YOLOv8", version: "blood-model", loaded: false },
        classificationModel: { name: "ResNet50", version: "wbc-model", loaded: false },
        inferenceEngine: { name: "PyTorch / OpenCV", online: false },
        apiOnline: false,
        frontendVersion: "1.0.0",
      };
    }
  },

  async dashboard(): Promise<DashboardStats> {
    if (USE_MOCK_DATA) {
      await delay(450);
      return {
        totalAnalyses: 0,
        successfulAnalyses: 0,
        recentReports: 0,
        averageProcessingTime: 0,
        cellDistribution: [
          { cellType: "RBC", count: 0 },
          { cellType: "WBC", count: 0 },
          { cellType: "Platelet", count: 0 },
        ],
        activity: [],
      };
    }

    const analyses = readLocalAnalyses();
    const totalAnalyses = analyses.length;
    const successfulAnalyses = analyses.filter((a) => a.status === "completed").length;
    const recentReports = totalAnalyses;
    const averageProcessingTime = analyses.length
      ? analyses.reduce((sum, a) => sum + (a.processingTime || 0), 0) / analyses.length
      : 0;

    const cellDistribution = [
      { cellType: "RBC" as const, count: analyses.reduce((sum, a) => sum + (a.cellStatistics.breakdown.find((b) => b.cellType === "RBC")?.count ?? 0), 0) },
      { cellType: "WBC" as const, count: analyses.reduce((sum, a) => sum + (a.cellStatistics.breakdown.find((b) => b.cellType === "WBC")?.count ?? 0), 0) },
      { cellType: "Platelet" as const, count: analyses.reduce((sum, a) => sum + (a.cellStatistics.breakdown.find((b) => b.cellType === "Platelet")?.count ?? 0), 0) },
    ];

    return {
      totalAnalyses,
      successfulAnalyses,
      recentReports,
      averageProcessingTime,
      cellDistribution,
      activity: [],
    };
  },

  async analyze(payload: AnalyzeRequest, onProgress?: (step: number) => void): Promise<Analysis> {
    if (USE_MOCK_DATA) {
      const fallback = "";
      for (let step = 0; step < 6; step += 1) {
        await delay(650);
        onProgress?.(step + 1);
      }
      const mock = toUiAnalysis({
        analysis_id: `ANL-${Math.floor(2100 + Math.random() * 800)}`,
        avg_confidence: 0.94,
        raw_counts: { rbc: 4820000, wbc: 7400, platelet: 248000 },
        calibrated_counts: { rbc: 4820000, wbc: 7400, platelet: 248000 },
        overall: "Normal",
        summary: ["Normal overall result", "Counts within reference range"],
        wbc_differential: { Neutrophil: 58, Lymphocyte: 29, Monocyte: 8, Eosinophil: 3, Basophil: 2 },
        detections: [],
      }, payload.file);
      const all = [...readLocalAnalyses(), mock];
      writeLocalAnalyses(all);
      return mock;
    }

    const form = new FormData();
    form.append("file", payload.file);
    if (payload.patientId) form.append("patient_id", payload.patientId);
    if (payload.sampleId) form.append("sample_id", payload.sampleId);
    if (payload.notes) form.append("notes", payload.notes);
    if (payload.gender) form.append("gender", payload.gender);
    if (payload.hb !== undefined) form.append("hb", String(payload.hb));
    if (payload.mcv !== undefined) form.append("mcv", String(payload.mcv));
    if (payload.mch !== undefined) form.append("mch", String(payload.mch));
    if (payload.mchc !== undefined) form.append("mchc", String(payload.mchc));
    if (payload.fusionMode) {
      form.append("fusion_mode", payload.fusionMode);
      form.append("prediction_mode", payload.fusionMode);
    }

    const backend = await request<any>(`/analyze?${new URLSearchParams({
      ...(payload.gender ? { gender: payload.gender } : {}),
      ...(payload.hb !== undefined ? { hb: String(payload.hb) } : {}),
      ...(payload.mcv !== undefined ? { mcv: String(payload.mcv) } : {}),
      ...(payload.mch !== undefined ? { mch: String(payload.mch) } : {}),
      ...(payload.mchc !== undefined ? { mchc: String(payload.mchc) } : {}),
      ...(payload.fusionMode ? { fusion_mode: payload.fusionMode, prediction_mode: payload.fusionMode } : {}),
    }).toString()}`, { method: "POST", body: form });
    
    const analysis = toUiAnalysis(backend, payload.file);
    const all = [...readLocalAnalyses(), analysis];
    writeLocalAnalyses(all);
    onProgress?.(6);
    return analysis;
  },

  async getAnalysis(id: string): Promise<Analysis> {
    if (USE_MOCK_DATA) {
      await delay(500);
      const current = readLocalAnalyses().find((item) => item.id === id);
      if (current) return current;
      return toUiAnalysis({ analysis_id: id, avg_confidence: 0.91, raw_counts: { rbc: 4800000, wbc: 7400, platelet: 250000 }, calibrated_counts: { rbc: 4800000, wbc: 7400, platelet: 250000 }, overall: "Normal", summary: ["Normal overall result"], wbc_differential: { Neutrophil: 58, Lymphocyte: 28, Monocyte: 6, Eosinophil: 4, Basophil: 4 }, detections: [] });
    }

    const saved = readLocalAnalyses().find((item) => item.id === id);
    if (saved) return saved;

    throw new ApiError("The requested resource could not be found.", 404);
  },

  async listAnalyses(): Promise<AnalysisSummary[]> {
    if (USE_MOCK_DATA) {
      await delay(500);
      return readLocalAnalyses().map((item) => ({
        id: item.id,
        createdAt: item.createdAt,
        imageUrl: item.imageUrl,
        imageName: item.imageName,
        cellsDetected: item.cellStatistics.totalDetected,
        riskLevel: item.anemiaRisk.level,
        confidence: item.confidence,
        status: item.status,
      }));
    }

    return readLocalAnalyses().map((item) => ({
      id: item.id,
      createdAt: item.createdAt,
      imageUrl: item.imageUrl,
      imageName: item.imageName,
      cellsDetected: item.cellStatistics.totalDetected,
      riskLevel: item.anemiaRisk.level,
      confidence: item.confidence,
      status: item.status,
    }));
  },

  async listReports(): Promise<Report[]> {
    return readLocalAnalyses()
      .filter((item) => item.status === "completed")
      .map((item) => ({
        id: `RPT-${item.id}`,
        analysisId: item.id,
        createdAt: item.createdAt,
        result: item.anemiaRisk?.explanation ?? "Analysis complete.",
        riskLevel: item.anemiaRisk?.level ?? "low",
        status: "ready" as const,
      }));
  },

  async getReport(id: string): Promise<Report> {
    const saved = readLocalAnalyses().find((item) => item.id === id);
    if (!saved) throw new ApiError("The requested resource could not be found.", 404);

    return {
      id: `RPT-${id}`,
      analysisId: id,
      createdAt: saved.createdAt,
      result: saved.anemiaRisk.explanation,
      riskLevel: saved.anemiaRisk.level,
      status: "ready",
    };
  },

  downloadReportUrl(id: string): string {
    return `${API_BASE_URL}/report/${encodeURIComponent(id)}`;
  },
};

export const queryKeys = {
  health: ["health"] as const,
  dashboard: ["dashboard"] as const,
  analyses: ["analyses"] as const,
  analysis: (id: string) => ["analysis", id] as const,
  reports: ["reports"] as const,
};
