import type { Project } from "./types";

export const defaultProject: Project = {
  name: "抓取搬运多段示例",
  waypoints: [
    [0.7761, -2.4, 1.6057],
    [1.1761, -2.1472, 1.9557],
    [1.3261, -2.1472, 2.1057],
  ],
  speed_limits: [0.8, 0.8, 0.8],
  segment_speed_limits: [
    [0.8, 0.8, 0.8],
    [0.8, 0.8, 0.8],
  ],
  joint_limits: [
    [-2.8, 2.8],
    [-2.4, 2.4],
    [-2.4, 2.4],
  ],
  link_lengths: [1.4, 1, 0.7],
  obstacles: [{ name: "shelf", x: 2.2, y: 1.6, radius: 0.1 }],
  carried_objects: [
    { name: "block", x: 1.49, y: 1.05, radius: 0.12, hold_dx: 0.12, hold_dy: 0 },
  ],
  actions: [
    { type: "grasp", waypoint_index: 1, object: "block" },
    { type: "release", waypoint_index: 2, object: "block" },
  ],
  link_radius: 0.08,
  sample_dt: 0.02,
  grasp_tolerance: 0.25,
  junction_speed_tolerance: 0.9,
};
