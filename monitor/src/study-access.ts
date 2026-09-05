import type { StudyJob } from "./types";

const key = "brain-study-history-v1";
export type SavedStudy = { jobId: string; accessToken: string; createdAt: string; expiresAt: string; state: string; modelId?: string };
export function history(): SavedStudy[] {
  try { return (JSON.parse(window.localStorage.getItem(key) || "[]") as SavedStudy[]).filter(row => /^[0-9a-f-]{36}$/.test(row.jobId) && /^[0-9a-f]{64}$/.test(row.accessToken) && Date.parse(row.expiresAt) > Date.now()); }
  catch { return []; }
}
export function remember(job: StudyJob & { accessToken?: string }) {
  const rows = history();
  const accessToken = job.accessToken || rows.find(row => row.jobId === job.jobId)?.accessToken;
  if (!accessToken) return;
  window.localStorage.setItem(key, JSON.stringify([{ jobId: job.jobId, accessToken, state: job.state, createdAt: job.createdAt, expiresAt: job.expiresAt, modelId: job.result?.provenance.model_id || "glioma-segresnet-20260828" }, ...rows.filter(row => row.jobId !== job.jobId)]));
}
export function forget(id: string) {
  window.localStorage.setItem(key, JSON.stringify(history().filter(row => row.jobId !== id)));
}
export function studyFetch(path: string, options: RequestInit = {}) {
  const id = path.match(/\/api\/studies\/([0-9a-f-]{36})/)?.[1];
  const token = history().find(row => row.jobId === id)?.accessToken;
  return fetch(path, { ...options, headers: { ...options.headers, ...(token ? { Authorization: `Bearer ${token}` } : {}) } });
}
export async function download(path: string, name: string) {
  const response = await studyFetch(path);
  if (!response.ok) throw new Error("Download unavailable. Reopen the study or check its expiry.");
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
