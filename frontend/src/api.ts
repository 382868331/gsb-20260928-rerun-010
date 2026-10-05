import type { Project, SavedProject, SavedTask, TaskDefinition, TaskResult, ValidationResult, WaitPlanResult } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ?? `请求失败：${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const validateProject = (project: Project) =>
  request<ValidationResult>("/api/validate", {
    method: "POST",
    body: JSON.stringify(project),
  });

export const validateProjectSegments = (project: Project, reviewedSegments: number[], previousDigest: string | null) =>
  request<ValidationResult>("/api/validate-segments", {
    method: "POST",
    body: JSON.stringify({ project, reviewed_segments: reviewedSegments, previous_validation_digest: previousDigest }),
  });

export const saveProject = (project: Project) =>
  request<SavedProject>("/api/projects", {
    method: "POST",
    body: JSON.stringify(project),
  });

export const updateProject = (id: string, project: Project) =>
  request<SavedProject>(`/api/projects/${id}`, {
    method: "PUT",
    body: JSON.stringify(project),
  });

export const listProjects = () => request<SavedProject[]>("/api/projects");

export const getProject = (id: string) => request<SavedProject>(`/api/projects/${id}`);

export const replayProject = (id: string) => request<ValidationResult>(`/api/projects/${id}/replay`);

export const validateTask = (task: TaskDefinition, previous: unknown | null) =>
  request<TaskResult>("/api/task/validate", {
    method: "POST",
    body: JSON.stringify({ task, previous }),
  });

export const planTaskWaits = (
  task: TaskDefinition,
  payload: { max_total_duration: number | null; allowed_seams: number[][]; immovable_segments: number[]; approve: boolean },
) =>
  request<WaitPlanResult>("/api/task/wait-plan", {
    method: "POST",
    body: JSON.stringify({ task, ...payload }),
  });

export const saveTask = (task: TaskDefinition) =>
  request<SavedTask>("/api/tasks", {
    method: "POST",
    body: JSON.stringify(task),
  });

export const updateTask = (id: string, task: TaskDefinition) =>
  request<SavedTask>(`/api/tasks/${id}`, {
    method: "PUT",
    body: JSON.stringify(task),
  });

export const listTasks = () => request<SavedTask[]>("/api/tasks");

export const getTask = (id: string) => request<SavedTask>(`/api/tasks/${id}`);
