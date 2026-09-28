export type Obstacle = {
  name: string;
  x: number;
  y: number;
  radius: number;
};

export type Project = {
  name: string;
  waypoints: number[][];
  speed_limits: number[];
  joint_limits: number[][];
  link_lengths: number[];
  obstacles: Obstacle[];
  link_radius: number;
  sample_dt: number;
};

export type SavedProject = Project & { id: string };

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
};

export type ValidationResult = {
  valid: boolean;
  duration: number;
  segment_durations: number[];
  frames: Frame[];
  collisions: Collision[];
  first_collision: Collision | null;
};
