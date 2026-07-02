import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Analysis, DatasetProfile, ModelSummary } from "../api/client";

type WorkspaceState = {
  currentDatasetId: string | null;
  datasetProfile: DatasetProfile | null;
  analysis: Analysis | null;
  trainedModel: ModelSummary | null;
  activeTrainingJobId: string | null;
  activeTrainingDatasetId: string | null;
  activeTrainingTarget: string | null;
  activeTrainingMode: string | null;
  setDataset: (profile: DatasetProfile) => void;
  setAnalysis: (analysis: Analysis) => void;
  setModel: (model: ModelSummary) => void;
  setTrainingJob: (jobId: string | null, datasetId?: string | null, target?: string | null, mode?: string | null) => void;
  loadDemo: () => void;
};

const demoAnalysis: Analysis = {
  dataset_id: "demo",
  rows: 891,
  columns: 12,
  schema: {
    type_counts: { numeric: 5, categorical: 4, boolean: 1, "id-like": 1, datetime: 1 },
    columns: [
      { name: "Age", inferred_type: "numeric", missing_percentage: 19.8, unique_count: 88, sample_values: ["22", "38"], warnings: [], target_like_candidate: true },
      { name: "PassengerClass", inferred_type: "categorical", missing_percentage: 0, unique_count: 3, sample_values: ["First", "Third"], warnings: [], target_like_candidate: true },
      { name: "Survived", inferred_type: "boolean", missing_percentage: 0, unique_count: 2, sample_values: ["0", "1"], warnings: [], target_like_candidate: true },
    ],
  },
  cleaning_report: { duplicate_rows: 0, missing_by_column: { Age: 177, Cabin: 687 } },
  eda_report: { dataset: { missing_values_total: 864, duplicate_rows: 0, type_counts: { numeric: 5, categorical: 4 } }, numeric: {}, categorical: {}, datetime: {} },
  feature_engineering_report: { count: 3, features: [] },
  rag_index_status: "ready",
  analysis_log: [
    { step: "Schema detected", status: "complete" },
    { step: "Cleaning report generated", status: "complete" },
    { step: "EDA completed", status: "complete" },
    { step: "Feature engineering completed", status: "complete" },
    { step: "Candidate charts generated", status: "complete" },
    { step: "Insights generated", status: "complete" },
    { step: "RAG index created", status: "complete" },
  ],
  chart_specs: [
    { chart_id: "age_distribution", title: "Age distribution", chart_type: "histogram", data: [{ bin: "0-16", count: 88 }, { bin: "17-32", count: 346 }, { bin: "33-48", count: 188 }, { bin: "49-64", count: 69 }], encoding: { x: "bin", y: "count" }, insight: "Most records cluster in young adult ranges, while older groups are thinner and should be interpreted with smaller sample sizes.", confidence: "medium", usefulness_score: 0.91, why: "Shows distribution shape and missingness impact.", render_diagnostics: { point_count: 4, estimated_payload_kb: 1, frontend_renderer: "echarts" } },
    { chart_id: "class_bar", title: "PassengerClass composition", chart_type: "bar", data: [{ value: "Third", count: 491 }, { value: "First", count: 216 }, { value: "Second", count: 184 }], encoding: { x: "value", y: "count" }, insight: "The dataset is weighted toward one class, so grouped comparisons should account for imbalance.", confidence: "high", usefulness_score: 0.82, render_diagnostics: { point_count: 3, estimated_payload_kb: 1, frontend_renderer: "echarts" } },
    { chart_id: "fare_age_scatter", title: "Fare vs Age", chart_type: "scatter", data: [{ x: 22, y: 7.25 }, { x: 38, y: 71.28 }, { x: 26, y: 7.92 }, { x: 35, y: 53.1 }], encoding: { x: "x", y: "y" }, insight: "Fare varies widely at similar ages, suggesting age alone is not enough to explain fare patterns.", confidence: "medium", usefulness_score: 0.78, render_diagnostics: { point_count: 4, estimated_payload_kb: 1, frontend_renderer: "echarts" } },
  ],
};

export const useWorkspace = create<WorkspaceState>()(persist((set) => ({
  currentDatasetId: null,
  datasetProfile: null,
  analysis: null,
  trainedModel: null,
  activeTrainingJobId: null,
  activeTrainingDatasetId: null,
  activeTrainingTarget: null,
  activeTrainingMode: null,
  setDataset: (profile) => set({ currentDatasetId: profile.dataset_id, datasetProfile: profile }),
  setAnalysis: (analysis) => set({ analysis, currentDatasetId: analysis.dataset_id }),
  setModel: (trainedModel) => set({ trainedModel }),
  setTrainingJob: (jobId, datasetId = null, target = null, mode = null) => set({ activeTrainingJobId: jobId, activeTrainingDatasetId: datasetId, activeTrainingTarget: target, activeTrainingMode: mode }),
  loadDemo: () => set({
    currentDatasetId: "demo",
    datasetProfile: { dataset_id: "demo", filename: "demo_passenger_workspace.csv", rows: 891, columns: 12, schema: demoAnalysis.schema, analysis_status: "Complete", analysis_log: demoAnalysis.analysis_log },
    analysis: demoAnalysis,
    trainedModel: null,
    activeTrainingJobId: null,
    activeTrainingDatasetId: null,
    activeTrainingTarget: null,
    activeTrainingMode: null,
  }),
}), {
  name: "datapilot-workspace",
  partialize: (state) => ({
    currentDatasetId: state.currentDatasetId,
    datasetProfile: state.datasetProfile,
    analysis: state.analysis,
    trainedModel: state.trainedModel,
    activeTrainingJobId: state.activeTrainingJobId,
    activeTrainingDatasetId: state.activeTrainingDatasetId,
    activeTrainingTarget: state.activeTrainingTarget,
    activeTrainingMode: state.activeTrainingMode,
  }),
}));
