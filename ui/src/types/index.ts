/**
 * Domain types for HemaVision AI.
 * These mirror the expected JSON shapes returned by the Python backend.
 */

export type AnalysisStatus = "completed" | "processing" | "review_required" | "failed";

export type CellType = "RBC" | "WBC" | "Platelet";

export type RangeStatus = "normal" | "low" | "high" | "unknown";

export type RiskLevel = "low" | "moderate" | "high";

/** A single object-detection result returned by the detection model. */
export interface Detection {
  id: string;
  cellType: CellType;
  confidence: number;
  /** Normalised bounding box (0-1) relative to image width/height. */
  bbox: { x: number; y: number; width: number; height: number };
}

/** Per-cell-type aggregate produced by the classification stage. */
export interface CellStatistic {
  cellType: CellType;
  count: number;
  averageConfidence: number;
  percentage: number;
}

/** Headline value for a cell type (e.g. RBC 4.82M). */
export interface CellSummary {
  cellType: CellType;
  value: string;
  unit: string;
  referenceRange: string;
  status: RangeStatus;
  confidence: number;
}

export interface CellStatistics {
  totalDetected: number;
  summary: CellSummary[];
  breakdown: CellStatistic[];
}

export interface MorphologyMetrics {
  averageCellArea: number;
  circularity: number;
  aspectRatio: number;
  meanIntensity: number;
  shapeVariation: "Low" | "Moderate" | "High";
  /** Histogram of cell-area distribution for charting. */
  areaDistribution: { bin: string; count: number }[];
  /** Radar-style morphology profile, values normalised 0-100. */
  profile: { metric: string; value: number }[];
}

export interface AnemiaRisk {
  /** 0-100 model-derived indication score. */
  score: number;
  level: RiskLevel;
  explanation: string;
  contributingFactors: { factor: string; weight: number }[];
}

export interface Explainability {
  attentionMapUrl?: string | undefined;
  note: string;
  confidenceByClass: { cellType: CellType; confidence: number }[];
  featureSummary: { feature: string; value: string }[];
}

export interface AnalysisFlag {
  cellType: string;
  label: string;
  value: number;
  unit: string;
  severity: string;
  condition: string;
  reference: string;
}

export interface AnemiaMlResponse {
  prediction: string;
  isAnemic: boolean;
  probability: number;
  confidence: number;
  modelName: string;
  modelVersion: string;
  status: string;
  message: string;
  cbcInputs?: {
    Gender: string;
    Hemoglobin: number;
    MCV: number;
    MCH: number;
    MCHC: number;
  };
}

export interface ImageAnemiaResponse {
  prediction: string;
  isAnemic: boolean;
  probability: number;
  confidence: number;
  modelName: string;
  modelVersion: string;
  status: string;
  message: string;
}

export interface FusionAnemiaResponse {
  fusionPrediction: string;
  isAnemic: boolean;
  fusionProbability: number;
  fusionConfidence: number;
  fusionStrategy: string;
  imageProbability: number;
  cbcProbability: number;
  modelVersion: string;
  status: string;
  message: string;
  fusionMode?: string | undefined;
  latentGateMean?: number | undefined;
  isExperimental?: boolean | undefined;
  limitationNotice?: string | undefined;
}

export interface Analysis {
  id: string;
  createdAt: string;
  imageUrl: string;
  processedImageUrl?: string | undefined;
  imageName: string;
  patientId?: string | undefined;
  sampleId?: string | undefined;
  notes?: string | undefined;
  processingTime: number;
  status: AnalysisStatus;
  confidence: number;
  detections: Detection[];
  cellStatistics: CellStatistics;
  morphology: MorphologyMetrics;
  anemiaRisk: AnemiaRisk;
  explainability: Explainability;
  rawCounts?: { rbc: number; wbc: number; platelet: number };
  calibratedCounts?: { rbc: number; wbc: number; platelet: number };
  wbcDifferential?: Record<string, number>;
  cellFlags?: AnalysisFlag[];
  diffFlags?: AnalysisFlag[];
  overall?: string;
  summary?: string[];
  reportAvailable?: boolean;
  reportUrl?: string;
  isEstimatedDiff?: boolean;
  anemiaMl?: AnemiaMlResponse | undefined;
  anemiaImage?: ImageAnemiaResponse | undefined;
  anemiaFusion?: FusionAnemiaResponse | undefined;
  cbcInputs?: {
    Gender?: string;
    Hemoglobin?: number;
    MCV?: number;
    MCH?: number;
    MCHC?: number;
  } | undefined;
}

/** Compact row shape used by list/table views. */
export interface AnalysisSummary {
  id: string;
  createdAt: string;
  imageUrl: string;
  imageName: string;
  cellsDetected: number;
  riskLevel: RiskLevel;
  confidence: number;
  status: AnalysisStatus;
}

export interface Report {
  id: string;
  analysisId: string;
  createdAt: string;
  result: string;
  riskLevel: RiskLevel;
  status: "ready" | "generating" | "failed";
}

export interface ModelStatus {
  detectionModel: { name: string; version: string; loaded: boolean };
  classificationModel: { name: string; version: string; loaded: boolean };
  inferenceEngine: { name: string; online: boolean };
  apiOnline: boolean;
  frontendVersion: string;
}

export interface DashboardStats {
  totalAnalyses: number;
  successfulAnalyses: number;
  recentReports: number;
  averageProcessingTime: number;
  cellDistribution: { cellType: CellType; count: number }[];
  activity: { date: string; analyses: number }[];
}

export interface AnalyzeRequest {
  file: File;
  patientId?: string | undefined;
  sampleId?: string | undefined;
  notes?: string | undefined;
  gender?: string | undefined;
  hb?: number | undefined;
  mcv?: number | undefined;
  mch?: number | undefined;
  mchc?: number | undefined;
  fusionMode?: 'default' | 'experimental_gmu' | string | undefined;
}
