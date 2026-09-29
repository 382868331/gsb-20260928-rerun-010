from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, sqrt


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class Link:
    start: Point
    end: Point
    radius: float


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float


def forward_kinematics(angles_deg: list[float], link_lengths: list[float]) -> list[Point]:
    points = [Point(0.0, 0.0)]
    x = y = theta = 0.0
    for angle_deg, length in zip(angles_deg, link_lengths):
        theta += radians(angle_deg)
        x += length * cos(theta)
        y += length * sin(theta)
        points.append(Point(x, y))
    return points


def links_for(points: list[Point], link_radius: float) -> list[Link]:
    return [Link(a, b, link_radius) for a, b in zip(points, points[1:])]


def _sub(a: Point, b: Point) -> Point:
    return Point(a.x - b.x, a.y - b.y)


def _dot(a: Point, b: Point) -> float:
    return a.x * b.x + a.y * b.y


def _cross(a: Point, b: Point) -> float:
    return a.x * b.y - a.y * b.x


def segment_distance(a: Point, b: Point, p: Point) -> float:
    ab = _sub(b, a)
    length_squared = _dot(ab, ab)
    if length_squared == 0.0:
        delta = _sub(p, a)
        return sqrt(_dot(delta, delta))
    ratio = max(0.0, min(1.0, _dot(_sub(p, a), ab) / length_squared))
    closest = Point(a.x + ab.x * ratio, a.y + ab.y * ratio)
    delta = _sub(p, closest)
    return sqrt(_dot(delta, delta))


def segment_segment_distance(first: Link, second: Link) -> float:
    endpoint_distances = [
        segment_distance(first.start, first.end, second.start),
        segment_distance(first.start, first.end, second.end),
        segment_distance(second.start, second.end, first.start),
        segment_distance(second.start, second.end, first.end),
    ]

    def signed_area(a: Point, b: Point, c: Point) -> float:
        return _cross(_sub(b, a), _sub(c, a))

    def on_segment(a: Point, b: Point, c: Point) -> bool:
        return (
            min(a.x, b.x) - 1e-9 <= c.x <= max(a.x, b.x) + 1e-9
            and min(a.y, b.y) - 1e-9 <= c.y <= max(a.y, b.y) + 1e-9
        )

    a, b = first.start, first.end
    c, d = second.start, second.end
    o1, o2 = signed_area(a, b, c), signed_area(a, b, d)
    o3, o4 = signed_area(c, d, a), signed_area(c, d, b)
    intersects = (
        (o1 == 0 and on_segment(a, b, c))
        or (o2 == 0 and on_segment(a, b, d))
        or (o3 == 0 and on_segment(c, d, a))
        or (o4 == 0 and on_segment(c, d, b))
        or ((o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0) and 0 not in (o1, o2, o3, o4))
    )
    return 0.0 if intersects else min(endpoint_distances)


def collision_report(
    angles_deg: list[float],
    link_lengths: list[float],
    link_radius: float,
    joint_limits: list[tuple[float, float]],
    obstacles: list[Obstacle],
) -> dict | None:
    if len(angles_deg) != len(link_lengths):
        return {"type": "configuration", "message": "关节数量与连杆数量不一致"}

    for index, (angle, (lower, upper)) in enumerate(zip(angles_deg, joint_limits)):
        if angle < lower or angle > upper:
            return {
                "type": "joint_limit",
                "joint": index,
                "value": angle,
                "lower": lower,
                "upper": upper,
                "message": f"关节 {index + 1} 为 {angle:.2f}°，超出限位 [{lower:.0f}°, {upper:.0f}°]",
            }

    points = forward_kinematics(angles_deg, link_lengths)
    links = links_for(points, link_radius)

    for link_index, link in enumerate(links):
        for obstacle_index, obstacle in enumerate(obstacles):
            distance = segment_distance(link.start, link.end, Point(obstacle.x, obstacle.y))
            clearance = distance - link.radius - obstacle.radius
            if clearance < 0:
                return {
                    "type": "obstacle",
                    "link": link_index,
                    "obstacle": obstacle_index,
                    "clearance": clearance,
                    "message": f"连杆 {link_index + 1} 在途中穿过障碍 {obstacle_index + 1}",
                }

    for first_index in range(len(links)):
        for second_index in range(first_index + 2, len(links)):
            distance = segment_segment_distance(links[first_index], links[second_index])
            clearance = distance - 2 * link_radius
            if clearance < -1e-9:
                return {
                    "type": "self_collision",
                    "first_link": first_index,
                    "second_link": second_index,
                    "clearance": clearance,
                    "message": f"连杆 {first_index + 1} 与连杆 {second_index + 1} 碰撞",
                }

    return None


def segment_duration(start: list[float], end: list[float], max_joint_speeds: list[float]) -> float:
    duration = 0.0
    for a, b, speed in zip(start, end, max_joint_speeds):
        if speed <= 0:
            raise ValueError("关节速度上限必须为正数")
        duration = max(duration, abs(b - a) / speed)
    return max(duration, 1e-6)


def interpolate(start: list[float], end: list[float], ratio: float) -> list[float]:
    return [a + (b - a) * ratio for a, b in zip(start, end)]


def _sample(angles: list[float], time: float, segment: int, link_lengths: list[float], link_radius: float) -> dict:
    points = forward_kinematics(angles, link_lengths)
    return {
        "time": round(time, 6),
        "segment": segment,
        "angles": [round(value, 6) for value in angles],
        "joints": [{"x": round(point.x, 6), "y": round(point.y, 6), "z": 0.0} for point in points],
        "links": [
            {
                "start": {"x": round(link.start.x, 6), "y": round(link.start.y, 6), "z": 0.0},
                "end": {"x": round(link.end.x, 6), "y": round(link.end.y, 6), "z": 0.0},
                "radius": link_radius,
            }
            for link in links_for(points, link_radius)
        ],
    }


def simulate(project: dict, playback_dt: float = 0.05, validation_step_deg: float = 1.0) -> dict:
    robot = project["robot"]
    waypoints = project["waypoints"]
    speeds = project["max_joint_speeds"]
    link_lengths = robot["link_lengths"]
    link_radius = robot["link_radius"]
    joint_limits = [tuple(item) for item in robot["joint_limits"]]
    obstacles = [Obstacle(**item) for item in project.get("obstacles", [])]

    if len(waypoints) < 2:
        raise ValueError("至少需要两个关节姿态")
    if any(len(point["angles"]) != len(link_lengths) for point in waypoints):
        raise ValueError("每个姿态都必须包含全部关节角度")
    if len(speeds) != len(link_lengths):
        raise ValueError("速度上限数量必须与关节数量一致")

    durations: list[float] = []
    boundary_times = [0.0]
    for current, following in zip(waypoints, waypoints[1:]):
        duration = segment_duration(current["angles"], following["angles"], speeds)
        durations.append(duration)
        boundary_times.append(boundary_times[-1] + duration)

    rejection = None
    collision_time = 0.0
    for segment_index, (current, following, duration) in enumerate(zip(waypoints, waypoints[1:], durations)):
        max_delta = max(abs(b - a) for a, b in zip(current["angles"], following["angles"]))
        steps = max(1, int(max(1.0, max_delta / validation_step_deg)))
        for step in range(steps + 1):
            ratio = step / steps
            angles = interpolate(current["angles"], following["angles"], ratio)
            collision = collision_report(angles, link_lengths, link_radius, joint_limits, obstacles)
            if collision is not None:
                rejection = collision
                collision_time = boundary_times[segment_index] + duration * ratio
                break
        if rejection is not None:
            break

    samples = []
    for segment_index, (current, following, duration) in enumerate(zip(waypoints, waypoints[1:], durations)):
        step_count = max(1, int(round(duration / playback_dt)))
        for step in range(step_count):
            ratio = step / step_count
            time = boundary_times[segment_index] + duration * ratio
            samples.append(_sample(interpolate(current["angles"], following["angles"], ratio), time, segment_index, link_lengths, link_radius))
    samples.append(_sample(waypoints[-1]["angles"], boundary_times[-1], len(durations) - 1, link_lengths, link_radius))

    if rejection is not None:
        rejection["time"] = round(collision_time, 6)
        collision_sample = min(samples, key=lambda sample: abs(sample["time"] - collision_time))
        rejection["angles"] = collision_sample["angles"]

    return {
        "accepted": rejection is None,
        "duration": round(boundary_times[-1], 6),
        "segment_durations": [round(value, 6) for value in durations],
        "samples": samples,
        "rejection": rejection,
    }


def default_project() -> dict:
    return {
        "name": "三自由度平面机械臂",
        "robot": {
            "link_lengths": [1.1, 1.1, 1.1],
            "link_radius": 0.08,
            "joint_limits": [[-170, 170], [-170, 170], [-170, 170]],
        },
        "max_joint_speeds": [60, 60, 60],
        "waypoints": [
            {"name": "起点", "angles": [-70, 20, 20]},
            {"name": "终点", "angles": [70, 20, 20]},
        ],
        "obstacles": [{"x": 1.8, "y": 0.0, "radius": 0.25}],
    }
