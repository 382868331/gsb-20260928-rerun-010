export type Obstacle = {
  name: string;
  x: number;
  y: number;
  radius: number;
};

export type CarriedObject = {
  id: string;
  name: string;
  radius: number;
  offset_x: number;
  offset_y: number;
};

export type CarryEvent = {
  waypoint_index: number;
  object_id: string;
};

export type Project = {
  name: string;
  waypoints: number[][];
  speed_limits: number[];
  segment_speed_limits: number[][] | null;
  acceleration_limits: number[] | null;
  velocity_transition_time: number;
  joint_limits: number[][];
  link_lengths: number[];
  obstacles: Obstacle[];
  carried_objects: CarriedObject[];
  pick_events: CarryEvent[];
  place_events: CarryEvent[];
  gripper_radius: number;
  constraint_version: string;
  link_radius: number;
  sample_dt: number;
  safety_resolution: number;
};

export type PlannedEvent = {
  type: "pick" | "place";
  waypoint_index: number;
  object_id: string;
  time: number;
};

export type Collision = {
  time: number;
  segment: number;
  kind: string;
  message: string;
};

export type DangerInterval = {
  segment: number;
  start_time: number;
  end_time: number;
  kinds: string[];
  messages: string[];
};

export type ReplayWindow = {
  start_time: number;
  end_time: number;
  segment: number;
  kinds: string[];
};

export type Frame = {
  time: number;
  joint_angles: number[];
  links: [number, number][];
  carried_object_id: string | null;
  object_position: [number, number] | null;
};

export type ValidationResult = {
  valid: boolean;
  duration: number;
  segment_durations: number[];
  frames: Frame[];
  planned_events: PlannedEvent[];
  constraint_version: string;
  safety_resolution: number;
  display_sample_dt: number;
  checked_segments: number[];
  collisions: Collision[];
  first_collision: Collision | null;
  danger_intervals: DangerInterval[];
  replay_window: ReplayWindow | null;
  certified_safe: boolean;
};

export type Revision = {
  revision_id: string;
  created_from_digest: string;
  constraint_version: string;
  checked_segments: number[];
  rechecked_clear_segments: number[];
  invalidated_segments: number[];
  danger_intervals: DangerInterval[];
  first_collision: Collision | null;
  project_snapshot: Project;
  previous_summary: Record<string, unknown> | null;
};

export type RecheckResponse = {
  checked_segments: number[];
  invalidated_segments: number[];
  validation: ValidationResult;
  revision: Revision;
};

export type SavedProject = Project & {
  id: string;
  executable: boolean;
  planned_events: PlannedEvent[];
  validation_digest: string;
  last_validation: ValidationResult | null;
  validation_error: string | null;
  revisions: Revision[];
};
