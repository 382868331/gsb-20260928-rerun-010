export type Obstacle = {
  name: string;
  x: number;
  y: number;
  radius: number;
};

export type CarriedObject = {
  name: string;
  x: number;
  y: number;
  radius: number;
  hold_dx: number;
  hold_dy: number;
};

export type GripperAction = {
  type: "grasp" | "release";
  waypoint_index: number;
  object: string;
};

export type Project = {
  name: string;
  waypoints: number[][];
  speed_limits: number[];
  segment_speed_limits?: number[][];
  joint_limits: number[][];
  link_lengths: number[];
  obstacles: Obstacle[];
  carried_objects: CarriedObject[];
  actions: GripperAction[];
  link_radius: number;
  sample_dt: number;
  grasp_tolerance: number;
  junction_speed_tolerance: number;
};

export type SavedProject = Project & {
  id: string;
  executable: boolean;
  constraint_version: string | null;
  grasp_events: { time: number; waypoint_index: number; object: string }[];
  release_events: { time: number; waypoint_index: number; object: string }[];
  segments_solved: boolean[];
  validated_duration: number | null;
};

export type Collision = {
  time: number;
  segment: number;
  kind: string;
  message: string;
};

export type Frame = {
  time: number;
  joint_angles: number[];
  links: [number, number][];
  objects?: { name: string; x: number; y: number; radius: number; state: "held" | "resting" }[];
};

export type ValidationResult = {
  valid: boolean;
  duration: number;
  segment_durations: number[];
  segment_velocities: number[][];
  segments_solved: boolean[];
  grasp_events: { time: number; waypoint_index: number; object: string }[];
  release_events: { time: number; waypoint_index: number; object: string }[];
  constraint_version: string;
  frames: Frame[];
  collisions: Collision[];
  first_collision: Collision | null;
};
