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

export type TaskOperation = {
  id: string;
  name: string;
  version: string;
  start_waypoint_index: number;
  end_waypoint_index: number;
  report_digest: string | null;
};

export type HandoffCondition = {
  id: string;
  waypoint_index: number;
  source_operation_id: string;
  target_operation_id: string;
  pose: number[] | null;
  carried_object_id: string | null;
  gripper_state: "open" | "closed";
  confidence: "confirmed" | "unknown";
};

export type TaskFailureEvent = {
  id: string;
  name: string;
  kind: "pick_failure" | "early_release";
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
  task_operations: TaskOperation[];
  handoff_conditions: HandoffCondition[];
  task_failure_events: TaskFailureEvent[];
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
  gripper_state?: "open" | "closed";
  carry_confidence?: "confirmed" | "unknown";
};

export type CarriedRange = {
  object_id: string;
  start_time: number;
  end_time: number;
  confidence: string;
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
  carried_ranges?: CarriedRange[];
  unknown_ranges?: Array<{ start_time: number; end_time: number; object_id: string | null; reason: string }>;
  task_review?: {
    operations: TaskOperation[];
    handoff_conditions: HandoffCondition[];
    segment_provenance: Array<{
      segment: number;
      operation_id: string | null;
      operation_version: string | null;
      operation_report_digest: string | null;
      incoming_handoff_id: string | null;
      handoff_condition: HandoffCondition | null;
      digest: string;
    }>;
    changed_segments: number[];
    revalidation_required_segments: number[];
    baseline_digest: string | null;
  };
  abnormal_scenarios?: Array<ValidationResult & {
    scenario: { id?: string; name?: string; kind: "pick_failure" | "early_release"; waypoint_index: number; object_id: string };
  }>;
};

export type AbnormalValidationResult = ValidationResult & {
  scenario: { id?: string; name?: string; kind: "pick_failure" | "early_release"; waypoint_index: number; object_id: string };
};

export type SavedProject = Project & {
  id: string;
  executable: boolean;
  planned_events: PlannedEvent[];
  validation_digest: string;
  last_validation: ValidationResult | null;
  validation_error: string | null;
};
