import type { Project } from "./types";

export const defaultProject: Project = {
  name: "中间碰撞示例",
  waypoints: [
    [-0.65, 0, 0],
    [0.65, 0, 0],
  ],
  speed_limits: [0.8, 0.8, 0.8],
  joint_limits: [
    [-2.8, 2.8],
    [-2.4, 2.4],
    [-2.4, 2.4],
  ],
  link_lengths: [1.4, 1, 0.7],
  obstacles: [{ name: "middle-sweep", x: 1.75, y: 0.12, radius: 0.1 }],
  link_radius: 0.08,
  sample_dt: 0.02,
};
