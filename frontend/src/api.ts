import type { ProjectSummary, RobotProject, SimulationResult } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? "请求失败：" + response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  defaultProject: () => request<RobotProject>("/api/default-project"),
  simulate: (project: RobotProject) =>
    request<SimulationResult>("/api/simulate", { method: "POST", body: JSON.stringify(project) }),
  list: () => request<ProjectSummary[]>("/api/projects"),
  save: (project: RobotProject) =>
    request<RobotProject>("/api/projects", { method: "POST", body: JSON.stringify(project) }),
  update: (project: RobotProject) =>
    request<RobotProject>("/api/projects/" + project.id, { method: "PUT", body: JSON.stringify(project) }),
  load: (id: string) => request<RobotProject>("/api/projects/" + id),
};
