import type { Project } from "./types";

export const defaultProject: Project = {
  name: "抓放连续动作示例",
  waypoints: [
    [-0.65, 0, 0],
    [-0.2, 0.05, 0],
    [0.45, 0.08, 0],
    [0.75, 0, 0],
  ],
  speed_limits: [0.8, 0.8, 0.8],
  segment_speed_limits: [
    [0.8, 0.8, 0.8],
    [0.45, 0.45, 0.45],
    [0.45, 0.45, 0.45],
  ],
  acceleration_limits: [1.2, 1.2, 1.2],
  velocity_transition_time: 0.2,
  joint_limits: [
    [-2.8, 2.8],
    [-2.4, 2.4],
    [-2.4, 2.4],
  ],
  link_lengths: [1.4, 1, 0.7],
  obstacles: [{ name: "middle-sweep", x: 1.75, y: 0.12, radius: 0.1 }],
  carried_objects: [{ id: "block-a", name: "方块A", radius: 0.1, offset_x: 0.12, offset_y: 0 }],
  pick_events: [{ waypoint_index: 1, object_id: "block-a" }],
  place_events: [{ waypoint_index: 3, object_id: "block-a" }],
  gripper_radius: 0.06,
  constraint_version: "1.1",
  link_radius: 0.08,
  sample_dt: 0.02,
  reviewed_segments: [],
  previous_validation_digest: null,
  baseline_snapshot: null,
  wait_duration: 0,
};
