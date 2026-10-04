import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, CheckCircle2, Pause, Play, RotateCcw, Save, ScanSearch } from "lucide-react";
import { getProject, getTask, listProjects, listTasks, saveProject, saveTask, updateProject, updateTask, validateProject, validateProjectSegments, validateTask } from "./api";
import { defaultProject } from "./defaults";
import RobotScene from "./RobotScene";
import type { CarryEvent, FailureSpec, Frame, Project, SavedProject, SavedTask, TaskDefinition, TaskFrame, TaskResult, ValidationResult } from "./types";

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
 const [replayWindow, setReplayWindow] = useState<{ start: number; end: number } | null>(null);
 const startedAt = useRef(0);
 const offset = useRef(0);

  const [task, setTask] = useState<TaskDefinition>({ name: "串联任务", segments: [], failures: [], handover_tolerance: 1e-6, constraint_version: "1.1" });
  const [taskResult, setTaskResult] = useState<TaskResult | null>(null);
  const [taskMode, setTaskMode] = useState(false);
  const [abnormalView, setAbnormalView] = useState(false);
  const [savedTasks, setSavedTasks] = useState<SavedTask[]>([]);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [waitCompare, setWaitCompare] = useState({ boundary: 0, short: 0.51, enough: 0.71 });
  const [waitComparison, setWaitComparison] = useState<{ label: string; wait: number; result: TaskResult }[] | null>(null);

  const activeTaskPlan = taskMode && taskResult ? (abnormalView && taskResult.abnormal ? taskResult.abnormal : taskResult) : null;
  const viewFrames: Frame[] = activeTaskPlan ? activeTaskPlan.frames : result?.frames ?? [];
  const viewDuration = activeTaskPlan ? activeTaskPlan.duration : result?.duration ?? 0;
  const frame: Frame | null = viewFrames[frameIndex] ?? null;
  const taskFirstHazard = activeTaskPlan?.first_hazard ?? null;
  const invalidTime = taskMode
    ? replayWindow?.end ?? taskFirstHazard?.end_time ?? taskFirstHazard?.start_time ?? null
    : result?.valid ? null : replayWindow?.end ?? result?.first_collision?.end_time ?? result?.first_collision?.time ?? null;
  const taskFrame = taskMode && frame ? (frame as TaskFrame) : null;
  const sceneProject = taskFrame ? task.segments[taskFrame.segment]?.operation ?? project : project;

  const validate = useCallback(async (nextProject = project, shouldPlay = false) => {
    setMessage("");
    try {
      const validation = await validateProject(nextProject);
      setResult(validation);
      setReplayWindow(null);
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
    listTasks().then(setSavedTasks).catch(() => setSavedTasks([]));
  }, []);

  useEffect(() => {
    if (!playing || viewFrames.length === 0) return;
    startedAt.current = performance.now();
    let animation = 0;
    const tick = () => {
      const elapsed = offset.current + (performance.now() - startedAt.current) / 1000;
      const cutoff = invalidTime ?? viewDuration;
      const nextIndex = viewFrames.findIndex((item) => item.time >= elapsed);
      setFrameIndex(nextIndex === -1 ? viewFrames.length - 1 : nextIndex);
      if (elapsed >= cutoff) {
        setPlaying(false);
        offset.current = cutoff;
        return;
      }
      animation = requestAnimationFrame(tick);
    };
    animation = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animation);
  }, [playing, viewFrames, viewDuration, invalidTime, replayWindow]);

  const seek = (index: number) => {
    const nextFrame = viewFrames[index];
    setPlaying(false);
    setReplayWindow(null);
    setFrameIndex(index);
    offset.current = nextFrame?.time ?? 0;
  };

  const replay = () => {
    const start = replayWindow?.start ?? 0;
    offset.current = start;
    const nextIndex = viewFrames.findIndex((frame) => frame.time >= start);
    setFrameIndex(nextIndex === -1 ? 0 : nextIndex);
    setPlaying(true);
  };

  const replayNearFirstHazard = () => {
    const window = taskMode ? taskResult?.replay_window : result?.replay_window;
    if (!window) return;
    const nextWindow = { start: window.start_time, end: window.end_time };
    setReplayWindow(nextWindow);
    offset.current = nextWindow.start;
    const nextIndex = viewFrames.findIndex((frame) => frame.time >= nextWindow.start);
    setFrameIndex(nextIndex === -1 ? 0 : nextIndex);
    setPlaying(true);
  };

  const replayFullMotion = () => {
    setReplayWindow(null);
    offset.current = 0;
    setFrameIndex(0);
    setPlaying(true);
  };

  const recheckChangedSegments = async () => {
    if (!result) return;
    const digest = `${JSON.stringify(project.waypoints)}:${JSON.stringify(project.segment_speed_limits)}`;
    const validation = await validateProjectSegments(
      { ...project, previous_validation_digest: digest },
      Array.from({ length: Math.max(0, project.waypoints.length - 1) }, (_, index) => index),
      digest,
    );
    setResult(validation);
    setReplayWindow(null);
    setFrameIndex(0);
    setMessage(validation.review.stale_segments.length ? "局部复核发现旧结论已失效" : "已按当前约束重新检查指定区间");
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

  const addTaskSegment = () => {
    setTask((current) => ({
      ...current,
      segments: [
        ...current.segments,
        { segment_id: `seg-${Date.now()}-${current.segments.length}`, name: project.name, operation: cloneProject(project) },
      ],
    }));
  };

  const replaceTaskSegment = (index: number) => {
    setTask((current) => {
      const segments = [...current.segments];
      segments[index] = { ...segments[index], name: project.name, operation: cloneProject(project) };
      return { ...current, segments };
    });
  };

  const removeTaskSegment = (index: number) => {
    setTask((current) => ({
      ...current,
      segments: current.segments.filter((_, itemIndex) => itemIndex !== index),
      failures: current.failures
        .filter((failure) => failure.segment_index !== index)
        .map((failure) => (failure.segment_index > index ? { ...failure, segment_index: failure.segment_index - 1 } : failure)),
    }));
  };

  const updateFailure = (index: number, patch: Partial<FailureSpec>) => {
    setTask((current) => ({
      ...current,
      failures: current.failures.map((failure, itemIndex) => (itemIndex === index ? { ...failure, ...patch } : failure)),
    }));
  };

  const runTaskValidation = async (nextTask: TaskDefinition = task) => {
    setMessage("");
    try {
      const validation = await validateTask(nextTask, taskResult?.conclusion_sources ?? null);
      setTaskResult(validation);
      setTaskMode(true);
      setAbnormalView(false);
      setReplayWindow(null);
      setFrameIndex(0);
      offset.current = 0;
      setPlaying(false);
      if (!validation.valid) {
        setMessage(`完整任务被拒绝：${validation.first_hazard?.message ?? "存在危险区间"}`);
      } else if (validation.stale_segments.length > 0) {
        setMessage(`以下段的旧结论需重新核对：${validation.stale_segments.map((item) => `段${item.segment + 1}`).join("、")}`);
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "任务校验失败");
    }
  };

  const persistTask = async () => {
    const saved = currentTaskId ? await updateTask(currentTaskId, task) : await saveTask(task);
    setCurrentTaskId(saved.id);
    setSavedTasks(await listTasks());
    if (saved.last_task_validation) {
      setTaskResult(saved.last_task_validation);
      setTaskMode(true);
    }
    setMessage(saved.executable ? `任务已保存并确认可执行：${saved.name}` : `任务已保存为草稿：${saved.name}（存在危险或未知区间）`);
  };

  const loadTask = async (id: string) => {
    const saved = await getTask(id);
    const { id: _id, executable: _executable, validation_digest: _digest, last_task_validation, validation_error: _error, ...definition } = saved;
    setCurrentTaskId(id);
    setTask(definition);
    if (last_task_validation) {
      setTaskResult(last_task_validation);
      setTaskMode(true);
      setAbnormalView(false);
      setReplayWindow(null);
      setFrameIndex(0);
      offset.current = 0;
    }
  };

  const updateSegmentWait = (index: number, field: "wait" | "wait_before" | "wait_after", value: number) => {
    setTask((current) => {
      const next: TaskDefinition = JSON.parse(JSON.stringify(current));
      next.segments[index][field] = Math.max(0, value);
      return next;
    });
  };

  const addWaitSegment = () => {
    setTask((current) => ({
      ...current,
      segments: [
        ...current.segments,
        { segment_id: `wait-${Date.now()}-${current.segments.length}`, name: "等待", operation: null, wait: 0.5 },
      ],
    }));
  };

  const conclusionLabel = (outcome: TaskResult) =>
    outcome.valid ? "通过" : outcome.unknown_segments.length > 0 ? "状态不足" : "拒绝";

  const compareWaitVariants = async () => {
    setMessage("");
    const boundary = Math.min(waitCompare.boundary, Math.max(0, task.segments.length - 2));
    const variants = [
      { label: "无等待", wait: 0 },
      { label: "短等待", wait: waitCompare.short },
      { label: "足够等待", wait: waitCompare.enough },
    ];
    try {
      const outcomes = [];
      for (const variant of variants) {
        const nextTask: TaskDefinition = JSON.parse(JSON.stringify(task));
        nextTask.segments[boundary].wait_after = variant.wait;
        const result = await validateTask(nextTask, null);
        outcomes.push({ ...variant, result });
      }
      setWaitComparison(outcomes);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "等待对比失败");
    }
  };

  const applyWaitToTask = (wait: number) => {
    const boundary = Math.min(waitCompare.boundary, Math.max(0, task.segments.length - 2));
    updateSegmentWait(boundary, "wait_after", wait);
  };

  const persistTaskAndReload = async () => {
    const saved = currentTaskId ? await updateTask(currentTaskId, task) : await saveTask(task);
    setCurrentTaskId(saved.id);
    setSavedTasks(await listTasks());
    await loadTask(saved.id);
    setMessage(saved.executable ? `任务已保存并重开校验通过：${saved.name}` : `任务已保存并重开校验：${saved.name}（仍不可执行）`);
  };

  const taskObjectOptions = useMemo(() => {
    const byId = new Map<string, string>();
    task.segments.forEach((segment) => (segment.operation?.carried_objects ?? []).forEach((item) => byId.set(item.id, item.name)));
    return Array.from(byId, ([id, name]) => ({ id, name }));
  }, [task.segments]);

  const collisionAtCurrent = useMemo(() => {
    if (!frame) return null;
    if (taskMode && taskResult) {
      return taskResult.hazards.find((hazard) => Math.abs(hazard.start_time - frame.time) < 0.02)?.message ?? null;
    }
    if (!result) return null;
    return result.collisions.find((collision) => Math.abs(collision.time - frame.time) < project.sample_dt)?.message ?? null;
  }, [frame, result, taskMode, taskResult, project.sample_dt]);

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
          {taskMode && taskResult && (
            <div className={`message ${taskResult.valid ? "" : "warning"}`}>
              <AlertTriangle size={15} />
              完整任务「{task.name}」：{taskResult.valid ? "各段与段间交接均安全" : "存在危险或未知区间"}
              {taskResult.comparison.first_concern_time != null &&
                `；首次危险/未知 ${taskResult.comparison.first_concern_time.toFixed(3)}s（${
                  taskResult.comparison.first_concern_source === "normal"
                    ? "正常方案"
                    : taskResult.comparison.first_concern_source === "abnormal"
                      ? "异常后续方案"
                      : "无法确认"
                }）`}
              {abnormalView && "（当前查看：异常后续方案）"}
              {taskResult.replay_window && (
                <button onClick={replayNearFirstHazard}>
                  <ScanSearch size={15} />重放首次危险区间{taskResult.replay_window.crosses_boundary ? "（跨段边界）" : ""}
                </button>
              )}
              {taskResult.abnormal && (
                <button onClick={() => { setAbnormalView(!abnormalView); setFrameIndex(0); offset.current = 0; setReplayWindow(null); }}>
                  {abnormalView ? "查看正常方案" : "查看异常后续方案"}
                </button>
              )}
              <button onClick={() => setTaskMode(false)}>返回单段视图</button>
            </div>
          )}
          <div className="status-row">
            {result?.valid ? (
              <span className="status ok"><CheckCircle2 size={17} />轨迹安全</span>
            ) : (
              <span className="status bad"><AlertTriangle size={17} />轨迹已拒绝</span>
            )}
            <span>总时长 {result?.duration.toFixed(2) ?? "--"}s</span>
            <span>约束 {result?.constraint_version ?? project.constraint_version}</span>
            <span>当前 {frame?.time.toFixed(2) ?? "--"}s</span>
            <span>安全检查步长 {result?.collision_check_dt.toFixed(3) ?? "--"}s / 显示 {project.sample_dt}s</span>
          </div>
          {result && !result.valid && result.replay_window && (
            <div className="message warning">
              <AlertTriangle size={15} />
              首次危险区间：S{result.replay_window.segment + 1}，{result.replay_window.start_time.toFixed(3)}s–{result.replay_window.end_time.toFixed(3)}s；该区间为保守包络结论，不能按采样点安全处理。
              <button onClick={replayNearFirstHazard}><ScanSearch size={15} />只重放附近</button>
              <button onClick={replayFullMotion}>全程重放</button>
            </div>
          )}
          {result && (
            <div className="message">
              已复核段：{result.review.reviewed_segments.length ? result.review.reviewed_segments.map((item) => `S${item + 1}`).join("、") : "全部轨迹"}
              {result.review.stale_segments.length ? `；失效段：${result.review.stale_segments.map((item) => `S${item + 1}`).join("、")}` : "；旧危险结论已由当前结果替代"}
              <button onClick={recheckChangedSegments}><RotateCcw size={15} />复核局部轨迹</button>
            </div>
          )}
          <RobotScene
            project={sceneProject}
            frame={frame}
            invalidTime={invalidTime}
            hazardWindow={taskMode
              ? taskFirstHazard ? { start_time: taskFirstHazard.start_time, end_time: taskFirstHazard.end_time } : null
              : result?.first_collision ? { start_time: result.first_collision.time, end_time: result.first_collision.end_time ?? result.first_collision.time } : null}
          />
          <div className="timeline">
            <button onClick={() => (playing ? setPlaying(false) : replayWindow ? replay() : replayFullMotion())}>
              {playing ? <Pause size={16} /> : <Play size={16} />}{playing ? "暂停" : "重放"}
            </button>
            <input
              type="range"
              min={0}
              max={Math.max(0, viewFrames.length - 1)}
              value={frameIndex}
              onChange={(event) => seek(Number(event.target.value))}
            />
          </div>
          {message && <div className="message"><AlertTriangle size={15} />{message}</div>}
          {collisionAtCurrent && <div className="message">当前采样点：{collisionAtCurrent}</div>}
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
          <h2>任务串联</h2>
          <label className="single-input">任务名
            <input value={task.name} onChange={(event) => setTask({ ...task, name: event.target.value })} />
          </label>
          {task.segments.map((segment, index) => {
            const segmentResult = taskResult?.segments[index];
            const stale = taskResult?.stale_segments.find((item) => item.segment === index);
            const handoverHazard = taskResult?.hazards.find(
              (hazard) => hazard.segment === index && hazard.kind.startsWith("handover"),
            );
            return (
              <div key={segment.segment_id}>
                <div className="limit-row">
                  <span>
                    段{index + 1} {segment.name}
                    {segment.operation === null ? "·等待段" : ""}
                    {segmentResult ? (segmentResult.valid ? "·安全" : "·危险") : "·未校验"}
                    {segmentResult ? ` ${segmentResult.start_time.toFixed(2)}–${segmentResult.end_time.toFixed(2)}s` : ""}
                    {stale ? "·需复核" : ""}
                    {index > 0 && (handoverHazard ? "·交接异常" : segmentResult?.handover_out ? "·交接一致" : "")}
                  </span>
                  {segment.operation !== null && (
                    <button onClick={() => replaceTaskSegment(index)}>替换为当前工程</button>
                  )}
                  <button className="icon" onClick={() => removeTaskSegment(index)}>×</button>
                </div>
                {segment.operation === null ? (
                  <label className="single-input">等待时长 s
                    <NumberInput value={segment.wait ?? 0} step={0.05} onChange={(next) => updateSegmentWait(index, "wait", next)} />
                  </label>
                ) : (
                  <div className="limit-row">
                    <label className="single-input">段前等待 s
                      <NumberInput value={segment.wait_before ?? 0} step={0.05} onChange={(next) => updateSegmentWait(index, "wait_before", next)} />
                    </label>
                    <label className="single-input">段后等待 s
                      <NumberInput value={segment.wait_after ?? 0} step={0.05} onChange={(next) => updateSegmentWait(index, "wait_after", next)} />
                    </label>
                  </div>
                )}
              </div>
            );
          })}
          <button onClick={addTaskSegment}>加入当前工程为一段</button>
          <button onClick={addWaitSegment}>新增独立等待段</button>
          <button className="primary" onClick={() => runTaskValidation()} disabled={task.segments.length < 2}>
            校验完整任务
          </button>
          {taskResult && taskResult.seam_checks.length > 0 && (
            <div className="message">
              接缝校验：
              {taskResult.seam_checks.map((seam, seamIndex) => (
                <div key={seamIndex}>
                  {seam.between_segment_ids.join(" → ")}：
                  {seam.status === "proven" ? "已证实可执行" : seam.status === "violated" ? "违反" : "状态未知"}
                  ，缝处 {seam.boundary_time.toFixed(2)}s
                  {typeof seam.wait_time === "number" && seam.wait_time > 0 ? `，等待 ${seam.wait_time.toFixed(2)}s` : ""}
                  ，携带 {seam.carried_object_id ?? "无"}
                  {seam.joints.filter((joint) => !joint.ok).map((joint) =>
                    `；关节${joint.joint} ${joint.from_velocity.toFixed(2)}→${joint.to_velocity.toFixed(2)} rad/s，减速到停再反向加速共需 ${joint.required_transition_time.toFixed(2)}s`
                    + (typeof seam.wait_time === "number" && seam.wait_time > 0
                      ? `（其中停留至少 ${(joint.required_dwell_time ?? joint.required_transition_time).toFixed(2)}s，过渡窗口 ${(joint.ramp_time ?? 0).toFixed(2)}s，实际等待 ${seam.wait_time.toFixed(2)}s）`
                      : ` / 允许 ${joint.allowed_transition_time.toFixed(2)}s`),
                  )}
                </div>
              ))}
            </div>
          )}
          {taskResult && (
            <div className="message">
              实际携带：
              {(abnormalView && taskResult.abnormal ? taskResult.abnormal.carried_ranges : taskResult.carried_ranges)
                .map((range) => `${range.object_id} 段${range.segment_index + 1} ${range.start_time.toFixed(2)}–${range.end_time.toFixed(2)}s`)
                .join("；") || "无"}
              {taskResult.unknown_intervals.length > 0 &&
                `；无法确认：${taskResult.unknown_intervals
                  .map((interval) => `${interval.start_time.toFixed(2)}s 起（段${interval.segment + 1}）`)
                  .join("、")}`}
            </div>
          )}

          <h2>等待对比</h2>
          <label className="single-input">对比接缝
            <select
              value={Math.min(waitCompare.boundary, Math.max(0, task.segments.length - 2))}
              onChange={(event) => setWaitCompare({ ...waitCompare, boundary: Number(event.target.value) })}
            >
              {task.segments.slice(0, -1).map((segment, index) => (
                <option key={segment.segment_id} value={index}>段{index + 1} → 段{index + 2}</option>
              ))}
            </select>
          </label>
          <div className="limit-row">
            <label className="single-input">短等待 s
              <NumberInput value={waitCompare.short} step={0.05} onChange={(next) => setWaitCompare({ ...waitCompare, short: next })} />
            </label>
            <label className="single-input">足够等待 s
              <NumberInput value={waitCompare.enough} step={0.05} onChange={(next) => setWaitCompare({ ...waitCompare, enough: next })} />
            </label>
          </div>
          <button onClick={compareWaitVariants} disabled={task.segments.length < 2}>
            对比无等待 / 短等待 / 足够等待
          </button>
          {waitComparison && (
            <div className="message">
              {waitComparison.map((entry) => (
                <div key={entry.label}>
                  {entry.label} {entry.wait.toFixed(2)}s：{conclusionLabel(entry.result)}
                  ，总时长 {entry.result.duration.toFixed(2)}s
                  {entry.result.first_hazard
                    ? `，首次违反 ${entry.result.first_hazard.start_time.toFixed(2)}–${entry.result.first_hazard.end_time.toFixed(2)}s`
                    : ""}
                  {entry.result.unknown_segments.length > 0
                    ? `，状态不足段 ${entry.result.unknown_segments.map((item) => item + 1).join("、")}`
                    : ""}
                </div>
              ))}
              <button onClick={() => applyWaitToTask(waitCompare.enough)}>应用足够等待到任务</button>
              <button onClick={persistTaskAndReload}>保存并重开后重新校验</button>
            </div>
          )}

          <h2>异常假设</h2>
          {task.failures.map((failure, index) => (
            <div className="limit-row" key={index}>
              <select value={failure.segment_index} onChange={(event) => updateFailure(index, { segment_index: Number(event.target.value) })}>
                {task.segments.map((segment, segmentIndex) => (
                  <option key={segment.segment_id} value={segmentIndex}>段{segmentIndex + 1}</option>
                ))}
              </select>
              <select value={failure.type} onChange={(event) => updateFailure(index, { type: event.target.value as FailureSpec["type"] })}>
                <option value="pick_failed">抓取失败</option>
                <option value="early_release">提前放开</option>
              </select>
              <select value={failure.object_id} onChange={(event) => updateFailure(index, { object_id: event.target.value })}>
                {taskObjectOptions.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
              </select>
              {failure.type === "early_release" && (
                <NumberInput
                  value={(failure.waypoint_index ?? 0) + 1}
                  step={1}
                  onChange={(next) => updateFailure(index, { waypoint_index: Math.max(0, Math.round(next) - 1) })}
                />
              )}
              <button className="icon" onClick={() => setTask({ ...task, failures: task.failures.filter((_, itemIndex) => itemIndex !== index) })}>×</button>
            </div>
          ))}
          <button onClick={() => {
            if (task.segments.length === 0 || taskObjectOptions.length === 0) return;
            setTask({
              ...task,
              failures: [...task.failures, { segment_index: 0, type: "pick_failed", object_id: taskObjectOptions[0].id, waypoint_index: null }],
            });
          }}>新增异常假设</button>

          <h2>任务库</h2>
          <button onClick={persistTask} disabled={task.segments.length < 2}><Save size={16} />保存任务</button>
          <select value={currentTaskId ?? ""} onChange={(event) => event.target.value && loadTask(event.target.value)}>
            <option value="">选择已保存任务</option>
            {savedTasks.map((item) => (
              <option key={item.id} value={item.id}>{item.name}{item.executable ? " [executable]" : " [draft]"}</option>
            ))}
          </select>

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
