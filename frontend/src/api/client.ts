import axios from "axios";

const http = axios.create({ baseURL: "" });

// ── Apps ─────────────────────────────────────────────────────────────────────

export interface AppConfig {
  app_name: string;
  collection: string;
  ingestion: Record<string, unknown>;
  embedding: Record<string, unknown>;
  retrieval: Record<string, unknown>;
  reranking: Record<string, unknown>;
  generation: Record<string, unknown>;
  defaults: Record<string, unknown>;
  tasks?: Record<string, unknown>;
  default_task?: string | null;
}

export async function listApps(): Promise<string[]> {
  const res = await http.get("/apps");
  return res.data.applications;
}

export async function getApp(name: string): Promise<AppConfig> {
  const res = await http.get(`/apps/${name}`);
  return res.data;
}

export async function registerApp(config: AppConfig): Promise<AppConfig> {
  const res = await http.post("/apps/register", config);
  return res.data.application;
}

export async function updateApp(
  name: string,
  config: Omit<AppConfig, "app_name">
): Promise<AppConfig> {
  const res = await http.put(`/apps/${name}`, config);
  return res.data.application;
}

export async function deleteApp(name: string): Promise<void> {
  await http.delete(`/apps/${name}`);
}

// ── Collections ───────────────────────────────────────────────────────────────

export interface CollectionInfo {
  name: string;
  points_count: number | null;
  vector_size: number | null;
  status: string;
}

export async function listCollections(): Promise<CollectionInfo[]> {
  const res = await http.get("/collections");
  return res.data.collections;
}

export async function getCollection(name: string): Promise<CollectionInfo> {
  const res = await http.get(`/collections/${name}`);
  return res.data;
}

export async function createCollection(
  name: string,
  vector_size: number
): Promise<void> {
  await http.post("/collections", { name, vector_size });
}

export async function deleteCollection(name: string): Promise<void> {
  await http.delete(`/collections/${name}`);
}

// ── Query / Generate ──────────────────────────────────────────────────────────

export interface QueryResponse {
  answer: unknown;
  sources: Array<{
    section: string;
    page: number | null;
    text?: string;
    score?: number;
    rerank_score?: number;
    hierarchy?: string;
  }>;
}

export async function query(
  app_name: string,
  user_id: string,
  queryText: string,
  task?: string,
  doc_ids?: string[]
): Promise<QueryResponse> {
  const res = await http.post("/query", {
    app_name,
    user_id,
    query: queryText,
    task: task || undefined,
    doc_ids: doc_ids && doc_ids.length > 0 ? doc_ids : undefined,
  });
  return res.data;
}
