import type { Project, SavedProject, ValidationResult } from "./types";

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
