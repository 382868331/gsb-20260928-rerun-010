import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, Pause, Play, RotateCcw, Save } from "lucide-react";
import { getProject, listProjects, saveProject, updateProject, validateProject } from "./api";
import { defaultProject } from "./defaults";
import RobotScene from "./RobotScene";
import type { CarryEvent, Frame, Project, SavedProject, ValidationResult } from "./types";

const cloneProject = (project: Project): Project => JSON.parse(JSON.stringify(project));

const segmentSpeeds = (project: Project) => {
  const segments = project.segment_speed_limits ?? project.waypoints.slice(1).map(() => project.speed_limits);
  return segments.length === project.waypoints.length - 1 ? segments : project.waypoints.slice(1).map(() => project.speed_limits);
};

const setSegmentSpeeds = (project: Project) => {
  project.segment_speed_limits = segmentSpeeds(project).map((values) => [...values]);
};

const shiftEventIndices = (events: CarryEvent[], removedIndex: number) =>
  events
    .filter((event) => event.waypoint_index !== removedIndex)
    .map((event) => ({
      ...event,
      waypoint_index: event.waypoint_index > removedIndex ? event.waypoint_index - 1 : event.waypoint_index,
    }));

function NumberInput({ value, onChange, step = 0.05 }: { value: number; onChange: (value: number) => void; step?: number }) {
  return (
    <input
      type="number"
      step={step}
      value={Number.isFinite(value) ? value : 0}
      onChange={(event) => onChange(Number(event.target.value))}
    />
  );
}

export default function App() {
  const [project, setProject] = useState<Project>(() => cloneProject(defaultProject));
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [message, setMessage] = useState<string>("");
  const [savedProjects, setSavedProjects] = useState<SavedProject[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const startedAt = useRef(0);
  const offset = useRef(0);

  const frame: Frame | null = result?.frames[frameIndex] ?? null;
  const invalidTime = result?.valid ? null : result?.first_collision?.time ?? null;

  const validate = useCallback(async (nextProject = project, shouldPlay = false) => {
    setMessage("");
    try {
      const validation = await validateProject(nextProject);
      setResult(validation);
      setFrameIndex(0);
      offset.current = 0;
      setPlaying(shouldPlay && validation.frames.length > 1);
      if (!validation.valid) {
        setMessage(`轨迹被拒绝：${validation.first_collision?.message ?? "发生碰撞"}`);
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "计算失败");
    }
  }, [project]);

  useEffect(() => {
    validate(project, false);
    listProjects().then(setSavedProjects).catch(() => setSavedProjects([]));
  }, []);

  useEffect(() => {
    if (!playing || !result) return;
    startedAt.current = performance.now();
    let animation = 0;
    const tick = () => {
      const elapsed = offset.current + (performance.now() - startedAt.current) / 1000;
      const cutoff = invalidTime ?? result.duration;
      const nextIndex = result.frames.findIndex((item) => item.time >= elapsed);
      setFrameIndex(nextIndex === -1 ? result.frames.length - 1 : nextIndex);
      if (elapsed >= cutoff) {
        setPlaying(false);
        offset.current = cutoff;
        return;
      }
      animation = requestAnimationFrame(tick);
    };
    animation = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animation);
  }, [playing, result, invalidTime]);

  const seek = (index: number) => {
    const nextFrame = result?.frames[index];
    setPlaying(false);
    setFrameIndex(index);
    offset.current = nextFrame?.time ?? 0;
  };

  const replay = () => {
    offset.current = 0;
    setFrameIndex(0);
    setPlaying(true);
  };

  const updateWaypoint = (pointIndex: number, jointIndex: number, value: number) => {
    setProject((current) => {
      const next = cloneProject(current);
      next.waypoints[pointIndex][jointIndex] = value;
      return next;
    });
  };

  const addWaypoint = () => {
    setProject((current) => {
      const next = cloneProject(current);
      const last = next.waypoints.at(-1) ?? next.link_lengths.map(() => 0);
      next.waypoints.push([...last]);
      setSegmentSpeeds(next);
      next.segment_speed_limits!.push(next.speed_limits.map((value) => value * 0.7));
      return next;
    });
  };

  const removeWaypoint = (index: number) => {
    setProject((current) => {
      if (current.waypoints.length <= 2) return current;
      const next = cloneProject(current);
      next.waypoints.splice(index, 1);
      setSegmentSpeeds(next);
      next.segment_speed_limits!.splice(index, 1);
      next.pick_events = shiftEventIndices(next.pick_events, index);
      next.place_events = shiftEventIndices(next.place_events, index);
      return next;
    });
  };

  const persist = async () => {
    const saved = currentId ? await updateProject(currentId, project) : await saveProject(project);
    setCurrentId(saved.id);
    setSavedProjects(await listProjects());
    setMessage(saved.executable ? `已保存并确认可执行：${saved.name}` : `已保存草稿：${saved.name}（不可执行，编辑内容已保留）`);
  };

  const load = async (id: string) => {
    const saved = await getProject(id);
    const { id: _id, ...nextProject } = saved;
    setCurrentId(id);
    setProject(nextProject);
    await validate(nextProject);
  };

  const collisionAtCurrent = useMemo(() => {
    if (!frame || !result) return null;
    return result.collisions.find((collision) => Math.abs(collision.time - frame.time) < project.sample_dt) ?? null;
  }, [frame, result, project.sample_dt]);

  return (
    <main className="app-shell">
      <header>
        <div>
          <p className="eyebrow">Offline Robot Programming</p>
          <h1>机械臂离线编程平台</h1>
        </div>
        <div className="header-actions">
          <input value={project.name} onChange={(event) => setProject({ ...project, name: event.target.value })} />
          <button onClick={() => validate(project)}><RotateCcw size={16} />重新计算</button>
          <button className="primary" onClick={persist}><Save size={16} />保存工程</button>
        </div>
      </header>

      <section className="workspace">
        <div className="viewer-card">
          <div className="status-row">
            {result?.valid ? (
              <span className="status ok"><CheckCircle2 size={17} />轨迹安全</span>
            ) : (
              <span className="status bad"><AlertTriangle size={17} />轨迹已拒绝</span>
            )}
            <span>总时长 {result?.duration.toFixed(2) ?? "--"}s</span>
            <span>约束 {result?.constraint_version ?? project.constraint_version}</span>
            <span>当前 {frame?.time.toFixed(2) ?? "--"}s</span>
          </div>
          <RobotScene project={project} frame={frame} invalidTime={invalidTime} />
          <div className="timeline">
            <button onClick={() => (playing ? setPlaying(false) : replay())}>
              {playing ? <Pause size={16} /> : <Play size={16} />}{playing ? "暂停" : "重放"}
            </button>
            <input
              type="range"
              min={0}
              max={Math.max(0, (result?.frames.length ?? 1) - 1)}
              value={frameIndex}
              onChange={(event) => seek(Number(event.target.value))}
            />
          </div>
          {message && <div className="message"><AlertTriangle size={15} />{message}</div>}
          {collisionAtCurrent && <div className="message">当前采样点：{collisionAtCurrent.message}</div>}
        </div>

        <aside className="editor">
          <h2>关节姿态</h2>
          <div className="table" role="table">
            <div className="row head"><span>点</span><span>关节1</span><span>关节2</span><span>关节3</span><span></span></div>
            {project.waypoints.map((point, pointIndex) => (
              <div className="row" key={pointIndex}>
                <span>P{pointIndex + 1}</span>
                {point.map((value, jointIndex) => (
                  <NumberInput key={jointIndex} value={value} onChange={(next) => updateWaypoint(pointIndex, jointIndex, next)} />
                ))}
                <button className="icon" onClick={() => removeWaypoint(pointIndex)} disabled={project.waypoints.length <= 2}>×</button>
              </div>
            ))}
          </div>
          <button onClick={addWaypoint}>新增姿态</button>

          <h2>各段速度上限 rad/s</h2>
          {segmentSpeeds(project).map((speeds, segmentIndex) => (
            <div className="limit-row" key={segmentIndex}>
              <span>S{segmentIndex + 1}</span>
              {speeds.map((value, jointIndex) => (
                <NumberInput key={jointIndex} value={value} step={0.05} onChange={(next) => {
                  const updated = cloneProject(project);
                  setSegmentSpeeds(updated);
                  updated.segment_speed_limits![segmentIndex][jointIndex] = next;
                  setProject(updated);
                }} />
              ))}
            </div>
          ))}

          <h2>速度衔接</h2>
          <label className="single-input">允许过渡时间 s
            <NumberInput value={project.velocity_transition_time} step={0.02} onChange={(next) => {
              setProject({ ...project, velocity_transition_time: next });
            }} />
          </label>
          {project.acceleration_limits?.map((value, index) => (
            <div className="limit-row" key={index}>
              <span>J{index + 1}</span>
              <NumberInput value={value} step={0.1} onChange={(next) => {
                const updated = cloneProject(project);
                updated.acceleration_limits![index] = next;
                setProject(updated);
              }} />
            </div>
          ))}

          <h2>抓放事件</h2>
          {project.pick_events.map((event, index) => (
            <div className="limit-row" key={`pick-${index}`}>
              <span>抓取 P{event.waypoint_index + 1}</span>
              <NumberInput value={event.waypoint_index + 1} step={1} onChange={(next) => {
                const updated = cloneProject(project);
                updated.pick_events[index].waypoint_index = Math.min(project.waypoints.length - 1, Math.max(0, next - 1));
                setProject(updated);
              }} />
              <select value={event.object_id} onChange={(change) => {
                const updated = cloneProject(project);
                updated.pick_events[index].object_id = change.target.value;
                setProject(updated);
              }}>
                {project.carried_objects.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
            </div>
          ))}
          {project.place_events.map((event, index) => (
            <div className="limit-row" key={`place-${index}`}>
              <span>放置 P{event.waypoint_index + 1}</span>
              <NumberInput value={event.waypoint_index + 1} step={1} onChange={(next) => {
                const updated = cloneProject(project);
                updated.place_events[index].waypoint_index = Math.min(project.waypoints.length - 1, Math.max(0, next - 1));
                setProject(updated);
              }} />
              <select value={event.object_id} onChange={(change) => {
                const updated = cloneProject(project);
                updated.place_events[index].object_id = change.target.value;
                setProject(updated);
              }}>
                {project.carried_objects.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
            </div>
          ))}
          <button onClick={() => {
            const updated = cloneProject(project);
            updated.pick_events.push({ waypoint_index: Math.max(0, updated.waypoints.length - 2), object_id: updated.carried_objects[0]?.id ?? "" });
            setProject(updated);
          }}>新增抓取</button>
          <button onClick={() => {
            const updated = cloneProject(project);
            updated.place_events.push({ waypoint_index: updated.waypoints.length - 1, object_id: updated.carried_objects[0]?.id ?? "" });
            setProject(updated);
          }}>新增放置</button>

          <h2>携带物体</h2>
          {project.carried_objects.map((item, index) => (
            <div className="obstacle-row" key={item.id}>
              <input value={item.name} onChange={(event) => {
                const updated = cloneProject(project);
                updated.carried_objects[index].name = event.target.value;
                setProject(updated);
              }} />
              <NumberInput value={item.radius} step={0.02} onChange={(next) => {
                const updated = cloneProject(project);
                updated.carried_objects[index].radius = next;
                setProject(updated);
              }} />
              <NumberInput value={item.offset_x} step={0.02} onChange={(next) => {
                const updated = cloneProject(project);
                updated.carried_objects[index].offset_x = next;
                setProject(updated);
              }} />
              <NumberInput value={item.offset_y} step={0.02} onChange={(next) => {
                const updated = cloneProject(project);
                updated.carried_objects[index].offset_y = next;
                setProject(updated);
              }} />
            </div>
          ))}

          <h2>速度上限 rad/s</h2>
          <div className="grid3">
            {project.speed_limits.map((value, index) => (
              <label key={index}>J{index + 1}
                <NumberInput value={value} onChange={(next) => {
                  const updated = cloneProject(project);
                  updated.speed_limits[index] = next;
                  setProject(updated);
                }} />
              </label>
            ))}
          </div>

          <h2>关节限位 rad</h2>
          {project.joint_limits.map((limit, index) => (
            <div className="limit-row" key={index}>
              <span>J{index + 1}</span>
              <NumberInput value={limit[0]} onChange={(next) => {
                const updated = cloneProject(project);
                updated.joint_limits[index][0] = next;
                setProject(updated);
              }} />
              <NumberInput value={limit[1]} onChange={(next) => {
                const updated = cloneProject(project);
                updated.joint_limits[index][1] = next;
                setProject(updated);
              }} />
            </div>
          ))}

          <h2>障碍物</h2>
          {project.obstacles.map((obstacle, index) => (
            <div className="obstacle-row" key={index}>
              <input value={obstacle.name} onChange={(event) => {
                const updated = cloneProject(project);
                updated.obstacles[index].name = event.target.value;
                setProject(updated);
              }} />
              <NumberInput value={obstacle.x} onChange={(next) => {
                const updated = cloneProject(project);
                updated.obstacles[index].x = next;
                setProject(updated);
              }} />
              <NumberInput value={obstacle.y} onChange={(next) => {
                const updated = cloneProject(project);
                updated.obstacles[index].y = next;
                setProject(updated);
              }} />
              <NumberInput value={obstacle.radius} step={0.02} onChange={(next) => {
                const updated = cloneProject(project);
                updated.obstacles[index].radius = next;
                setProject(updated);
              }} />
            </div>
          ))}

          <h2>工程库</h2>
          <select value={currentId ?? ""} onChange={(event) => event.target.value && load(event.target.value)}>
            <option value="">选择已保存工程</option>
            {savedProjects.map((item) => (
              <option key={item.id} value={item.id}>{item.name}{item.executable ? " [executable]" : " [draft]"}</option>
            ))}
          </select>
        </aside>
      </section>
    </main>
  );
}
