export type Vec2 = { x: number; y: number; z?: number };
export type Waypoint = { name: string; angles: number[] };
export type Obstacle = { x: number; y: number; radius: number };

export type RobotProject = {
  id?: string;
  name: string;
  robot: { link_lengths: number[]; link_radius: number; joint_limits: [number, number][] };
  max_joint_speeds: number[];
  waypoints: Waypoint[];
  obstacles: Obstacle[];
};

export type LinkPose = { start: Vec2; end: Vec2; radius: number };
export type TrajectorySample = {
  time: number;
  segment: number;
  angles: number[];
  joints: Vec2[];
  links: LinkPose[];
};
export type Rejection = {
  type: string;
  message: string;
  time: number;
  angles: number[];
  link?: number;
  obstacle?: number;
  joint?: number;
  clearance?: number;
} | null;
export type SimulationResult = {
  accepted: boolean;
  duration: number;
  segment_durations: number[];
  samples: TrajectorySample[];
  rejection: Rejection;
};
export type ProjectSummary = { id: string; name: string };
