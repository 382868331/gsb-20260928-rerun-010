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
  reviewed_segments: number[];
  previous_validation_digest: string | null;
  baseline_snapshot: ValidationResult | null;
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
  end_time?: number;
  confidence?: string;
};

export type HazardInterval = Collision & {
  start_time: number;
  end_time: number;
  segment: number;
  kind: string;
  message: string;
  confidence: string;
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
  collisions: Collision[];
  first_collision: Collision | null;
  collision_check_dt: number;
  display_sample_dt: number;
  hazard_intervals: HazardInterval[];
  replay_window: { start_time: number; end_time: number; segment: number } | null;
  review: {
    previous_validation_digest: string | null;
    reviewed_segments: number[];
    affected_segments: number[];
    stale_segments: number[];
    fully_rechecked: boolean;
  };
};

export type SavedProject = Project & {
  id: string;
  executable: boolean;
  planned_events: PlannedEvent[];
  validation_digest: string;
  last_validation: ValidationResult | null;
  validation_error: string | null;
};

export type TaskSegment = {
  segment_id: string;
  name: string;
  operation: Project | null;
  wait?: number;
  wait_before?: number;
  wait_after?: number;
};

export type FailureSpec = {
  segment_index: number;
  type: "pick_failed" | "early_release";
  object_id: string;
  waypoint_index?: number | null;
};

export type TaskDefinition = {
  name: string;
  segments: TaskSegment[];
  failures: FailureSpec[];
  handover_tolerance: number;
  constraint_version: string;
};

export type TaskFrame = Frame & {
  segment: number;
  local_time: number;
};

export type TaskHazard = {
  start_time: number;
  end_time: number;
  segment: number;
  inner_segment: number | null;
  kind: string;
  message: string;
  confidence: string;
  detail?: Record<string, unknown>;
};

export type HandoverState = {
  joint_angles: number[];
  carried_object_id: string | null;
};

export type TaskSegmentResult = {
  segment_id: string;
  name: string;
  digest: string;
  kind?: "motion" | "wait";
  start_time: number;
  end_time: number;
  wait?: number;
  wait_before?: number;
  wait_after?: number;
  handover_in: HandoverState;
  handover_out: HandoverState | null;
  status?: "proven" | "violated" | "unknown";
  valid: boolean;
  error: string | null;
  report: ValidationResult | null;
  entry_velocity?: number[] | null;
  exit_velocity?: number[] | null;
};

export type SeamJointCheck = {
  joint: number;
  from_velocity: number;
  to_velocity: number;
  velocity_change: number;
  required_transition_time: number;
  allowed_transition_time: number;
  acceleration_limit: number;
  ok: boolean;
};

export type SeamCheck = {
  between_segments: number[];
  between_segment_ids: string[];
  between_names?: string[];
  boundary_time: number;
  wait_time?: number;
  transition_window_start?: number;
  transition_window_end?: number;
  status: "proven" | "violated" | "unknown";
  confidence: string;
  carried_object_id: string | null;
  joint_angles?: number[];
  joints: SeamJointCheck[];
  reason?: string;
};

export type CarriedRange = {
  object_id: string;
  segment_index: number;
  start_time: number;
  end_time: number;
};

export type UnknownInterval = {
  start_time: number;
  end_time: number;
  segment: number;
  kind: string;
  message: string;
  confidence: string;
};

export type TaskReplayWindow = {
  start_time: number;
  end_time: number;
  segment: number;
  crosses_boundary: boolean;
  boundaries: number[];
};

export type TaskPlanResult = {
  valid: boolean;
  duration: number;
  segments: Omit<TaskSegmentResult, "report">[];
  hazards: TaskHazard[];
  first_hazard: TaskHazard | null;
  frames: TaskFrame[];
  carried_ranges: CarriedRange[];
  seam_checks: SeamCheck[];
  unknown_segments: number[];
  replay_window: TaskReplayWindow | null;
};

export type TaskResult = {
  valid: boolean;
  duration: number;
  constraint_version: string;
  handover_tolerance: number;
  segments: TaskSegmentResult[];
  hazards: TaskHazard[];
  first_hazard: TaskHazard | null;
  replay_window: TaskReplayWindow | null;
  frames: TaskFrame[];
  carried_ranges: CarriedRange[];
  boundaries: number[];
  seam_checks: SeamCheck[];
  unknown_segments: number[];
  stale_segments: { segment: number; reason: string }[];
  recheck_required: number[];
  conclusion_sources: unknown;
  abnormal: TaskPlanResult | null;
  unknown_intervals: UnknownInterval[];
  comparison: {
    normal_first_hazard: TaskHazard | null;
    abnormal_first_hazard: TaskHazard | null;
    first_unknown: UnknownInterval | null;
    first_concern_time: number | null;
    first_concern_source: "normal" | "abnormal" | "unknown" | null;
    first_concern: TaskHazard | UnknownInterval | null;
  };
  collision_check_dt: number;
};

export type SavedTask = TaskDefinition & {
  id: string;
  executable: boolean;
  validation_digest: string;
  last_task_validation: TaskResult | null;
  validation_error: string | null;
};
