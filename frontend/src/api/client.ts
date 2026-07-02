export const API_BASE = "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<T>;
}

export type DatasetProfile = {
  dataset_id: string;
  filename: string;
  rows: number;
  columns: number;
  schema: { columns: SchemaColumn[]; type_counts: Record<string, number> };
  analysis_status: string;
  analysis_log?: LogItem[];
  trained_model_id?: string;
};

export type SchemaColumn = {
  name: string;
  inferred_type: string;
  missing_percentage: number;
  unique_count: number;
  sample_values: string[];
  warnings: string[];
  target_like_candidate: boolean;
};

export type LogItem = { step: string; status: string };
export type ChartSpec = {
  chart_id: string;
  title: string;
  chart_type: string;
  data: Record<string, unknown>[];
  encoding: Record<string, string>;
  insight: string;
  confidence?: string;
  caveat?: string;
  why?: string;
  usefulness_score: number;
  excluded_missing_count?: number;
  excluded_missing_percent?: number;
  notes?: string[];
  computed_stats?: Record<string, unknown>;
  intent?: string;
  columns_used?: string[];
  reason_selected?: string;
  missing_values_excluded?: number;
  caveats?: string[];
  render_diagnostics: Record<string, unknown>;
};

export type Analysis = {
  dataset_id: string;
  rows: number;
  columns: number;
  schema: DatasetProfile["schema"];
  cleaning_report: { duplicate_rows: number; missing_by_column: Record<string, number> };
  eda_report: {
    dataset: Record<string, unknown>;
    numeric: Record<string, unknown>;
    categorical: Record<string, unknown>;
    datetime: Record<string, unknown>;
  };
  feature_engineering_report: { count: number; features: unknown[] };
  chart_specs: ChartSpec[];
  analysis_log: LogItem[];
  rag_index_status: string;
};

export type ModelSummary = {
  model_id: string;
  target: string;
  task_type: string;
  selected_model: string;
  mode: string;
  metrics: Record<string, number | null>;
  selected_metric?: string;
  baseline?: Record<string, unknown>;
  beats_baseline?: boolean | null;
  compared_models: unknown[];
  feature_schema: {
    features: ModelFeature[];
    excluded_features?: ModelFeature[];
    leakage_warnings?: { column: string; reason: string; action: string }[];
  };
  excluded_columns?: ModelFeature[];
  leakage_warnings?: { column: string; reason: string; action: string }[];
  warnings?: string[];
  validation_method?: string;
  train_rows?: number;
  test_rows?: number;
  class_balance?: Record<string, number> | null;
  random_state?: number;
  chat_prediction_enabled?: boolean;
  training_time_seconds: number;
  rows_used: number;
  features_used: number;
};

export type TrainingLog = { time: string; level: "info" | "warning" | "error" | "success"; message: string };
export type TrainingStep = { name: string; status: "complete" | "running" | "pending" | "failed"; started_at?: string | null; completed_at?: string | null; duration_seconds?: number | null };
export type TrainingModelStatus = { name: string; status: "pending" | "running" | "complete" | "failed" | "skipped" | "timeout"; started_at?: string | null; completed_at?: string | null; duration_seconds?: number | null; metrics?: Record<string, unknown> | null; error?: string | null };
export type TrainingJob = {
  job_id: string;
  dataset_id: string;
  target?: string;
  mode?: string;
  status: "queued" | "running" | "complete" | "failed" | "cancelled";
  progress: number;
  current_step: string;
  current_model: string | null;
  elapsed_seconds: number;
  estimated_remaining_seconds: number | null;
  logs: TrainingLog[];
  steps: TrainingStep[];
  models: TrainingModelStatus[];
  model_id: string | null;
  error: string | null;
};

export type ModelFeature = {
  feature_name: string;
  name?: string;
  display_name: string;
  type: "numeric" | "categorical" | "boolean" | "date" | string;
  allowed_values?: string[] | null;
  categories?: string[] | null;
  required: boolean;
  source: string;
  excluded_reason?: string | null;
};

export type ChatPredictResponse = {
  status: string;
  assistant_message: string;
  conversation_state: Record<string, unknown>;
  extracted_values?: Record<string, unknown>;
  missing_features?: ModelFeature[];
  prediction?: Record<string, unknown>;
};

export type ToolTrace = {
  user_question: string;
  route: "dataframe_tool" | "rag" | "hybrid" | "fallback" | string;
  detected_intent: string;
  resolved_columns: Record<string, unknown>;
  extracted_filters: unknown[];
  planned_tool_call: Record<string, unknown> | null;
  actual_tool_called: string | null;
  tool_result: Record<string, unknown> | null;
  rag_used: boolean;
  errors: string[];
  [key: string]: unknown;
};

export const api = {
  async upload(file: File) {
    const body = new FormData();
    body.append("file", file);
    return request<DatasetProfile>("/api/datasets/upload", { method: "POST", body });
  },
  dataset: (id: string) => request<DatasetProfile>(`/api/datasets/${id}`),
  analyze: (id: string) => request<Analysis>(`/api/datasets/${id}/analyze`, { method: "POST" }),
  analysis: (id: string) => request<Analysis>(`/api/datasets/${id}/analysis`),
  chat: (id: string, question: string) => request<{ answer: string; evidence: { title: string; text: string }[]; debug_evidence?: { title: string; text: string }[]; tool_trace?: ToolTrace }>(`/api/datasets/${id}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) }),
  train: (id: string, target: string, mode: string) => request<{ job_id: string; status: string }>(`/api/datasets/${id}/models/train`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target, mode }) }),
  trainingJob: (jobId: string) => request<TrainingJob>(`/api/training-jobs/${jobId}`),
  cancelTrainingJob: (jobId: string) => request<{ job_id: string; status: string }>(`/api/training-jobs/${jobId}/cancel`, { method: "POST" }),
  model: (id: string, modelId: string) => request<ModelSummary>(`/api/datasets/${id}/models/${modelId}`),
  predict: (id: string, modelId: string, values: Record<string, unknown>) => request<Record<string, unknown>>(`/api/datasets/${id}/models/${modelId}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ values }) }),
  chatPredict: (id: string, modelId: string, message: string, conversationState: Record<string, unknown>) => request<ChatPredictResponse>(`/api/datasets/${id}/models/${modelId}/chat-predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, conversation_state: conversationState }) }),
};

export function downloadUrl(path: string) {
  window.location.href = path;
}
