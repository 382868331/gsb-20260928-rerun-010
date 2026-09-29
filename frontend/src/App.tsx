import { useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play, Plus, RotateCcw, Save, Trash2 } from "lucide-react";
import { api } from "./api";
import Scene from "./Scene";
import type { ProjectSummary, RobotProject, SimulationResult, TrajectorySample } from "./types";

const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value));

export default function App() {
  const [project, setProject] = useState<RobotProject | null>(null);
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [savedProjects, setSavedProjects] = useState<ProjectSummary[]>([]);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(true);
  const animationRef = useRef<number | null>(null);
  const startedAtRef = useRef(0);
  const startFrameRef = useRef(0);

  const samples = result?.samples ?? [];
  const currentSample: TrajectorySample | null = samples[frame] ?? null;
  const rejectionSample = useMemo(() => {
    if (!result?.rejection) return null;
    const rejectionTime = result.rejection.time;
    return samples.reduce((closest, sample) =>
      Math.abs(sample.time - rejectionTime) < Math.abs(closest.time - rejectionTime) ? sample : closest
    );
  }, [result, samples]);

  async function refreshProjects() {
    setSavedProjects(await api.list());
  }

  async function runSimulation(nextProject: RobotProject, autoplay = false) {
    setMessage("");
    try {
      const nextResult = await api.simulate(nextProject);
      setResult(nextResult);
      setFrame(0);
      setPlaying(autoplay && nextResult.accepted);
      if (!nextResult.accepted && nextResult.rejection) {
        setMessage("轨迹被拒绝：" + nextResult.rejection.message);
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "仿真失败");
      setPlaying(false);
    }
  }

  useEffect(() => {
    Promise.all([api.defaultProject(), refreshProjects()])
      .then(([initial]) => {
        setProject(initial);
        return runSimulation(initial);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!playing || samples.length < 2) return;
    startedAtRef.current = performance.now();
    startFrameRef.current = frame;
    const startSampleTime = samples[frame]?.time ?? 0;
    const finalTime = samples[samples.length - 1].time;

    const tick = (now: number) => {
      const elapsed = (now - startedAtRef.current) / 1000;
      const targetTime = Math.min(startSampleTime + elapsed, finalTime);
      let nextFrame = startFrameRef.current;
      while (nextFrame + 1 < samples.length && samples[nextFrame + 1].time <= targetTime) nextFrame += 1;
      setFrame(nextFrame);
      if (targetTime >= finalTime) {
        setPlaying(false);
        return;
      }
      animationRef.current = requestAnimationFrame(tick);
    };
    animationRef.current = requestAnimationFrame(tick);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [playing]);

  if (loading || !project) return <main className="loading">正在加载机器人模型...</main>;

  const updateProject = (recipe: (draft: RobotProject) => void) => {
    const next = clone(project);
    recipe(next);
    setProject(next);
  };

  const setAngle = (waypointIndex: number, jointIndex: number, value: number) => {
    updateProject((draft) => {
      draft.waypoints[waypointIndex].angles[jointIndex] = value;
    });
  };

  const addWaypoint = () => {
    updateProject((draft) => {
      const previous = draft.waypoints[draft.waypoints.length - 1];
      draft.waypoints.push({
        name: "姿态 " + (draft.waypoints.length + 1),
        angles: [...previous.angles],
      });
    });
  };

  const removeWaypoint = (index: number) => {
    updateProject((draft) => {
      if (draft.waypoints.length > 2) draft.waypoints.splice(index, 1);
    });
  };

  const saveProject = async () => {
    const saved = project.id ? await api.update(project) : await api.save(project);
    setProject(saved);
    await refreshProjects();
    setMessage("工程已保存：" + saved.name);
  };

  const loadProject = async (id: string) => {
    if (!id) return;
    const loaded = await api.load(id);
    setProject(loaded);
    setPlaying(false);
    await runSimulation(loaded);
  };

  const sliderTime = currentSample?.time ?? 0;
  const jointCount = project.robot.link_lengths.length;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>机械臂离线编程平台</h1>
          <p>关节空间轨迹 · 中间帧限位与碰撞验证 · 不连接真实机械臂</p>
        </div>
        <div className="top-actions">
          <select value={project.id ?? ""} onChange={(event) => loadProject(event.target.value)}>
            <option value="">读取已保存工程</option>
            {savedProjects.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select>
          <button onClick={saveProject}><Save size={16} />保存工程</button>
        </div>
      </header>

      <section className="workspace">
        <aside className="panel editor">
          <label className="project-name">工程名称
            <input value={project.name} onChange={(event) => updateProject((draft) => { draft.name = event.target.value; })} />
          </label>

          <div className="robot-summary">
            <span>{jointCount} 个旋转关节</span>
            <span>限位：±170°</span>
            <span>连杆半径：{project.robot.link_radius} m</span>
          </div>

          <h2>关节速度上限（°/s）</h2>
          <div className="speed-grid">
            {project.max_joint_speeds.map((speed, index) => (
              <label key={index}>J{index + 1}
                <input type="number" min={1} step={1} value={speed}
                  onChange={(event) => updateProject((draft) => {
                    draft.max_joint_speeds[index] = Number(event.target.value);
                  })} />
              </label>
            ))}
          </div>

          <div className="section-title">
            <h2>关节姿态</h2>
            <button className="ghost" onClick={addWaypoint}><Plus size={15} />新增</button>
          </div>
          <div className="waypoints">
            {project.waypoints.map((waypoint, waypointIndex) => (
              <article className="waypoint-card" key={waypointIndex}>
                <header>
                  <input className="waypoint-name" value={waypoint.name}
                    onChange={(event) => updateProject((draft) => {
                      draft.waypoints[waypointIndex].name = event.target.value;
                    })} />
                  <button className="icon danger" disabled={project.waypoints.length <= 2}
                    onClick={() => removeWaypoint(waypointIndex)}><Trash2 size={15} /></button>
                </header>
                <div className="angle-grid">
                  {waypoint.angles.map((angle, jointIndex) => (
                    <label key={jointIndex}>J{jointIndex + 1}
                      <input type="number" value={angle}
                        onChange={(event) => setAngle(waypointIndex, jointIndex, Number(event.target.value))} />
                    </label>
                  ))}
                </div>
              </article>
            ))}
          </div>

          <h2>障碍物（圆盘）</h2>
          {project.obstacles.map((obstacle, index) => (
            <div className="obstacle-row" key={index}>
              <label>X<input type="number" step="0.1" value={obstacle.x}
                onChange={(event) => updateProject((draft) => { draft.obstacles[index].x = Number(event.target.value); })} /></label>
              <label>Y<input type="number" step="0.1" value={obstacle.y}
                onChange={(event) => updateProject((draft) => { draft.obstacles[index].y = Number(event.target.value); })} /></label>
              <label>R<input type="number" step="0.05" min="0.05" value={obstacle.radius}
                onChange={(event) => updateProject((draft) => { draft.obstacles[index].radius = Number(event.target.value); })} /></label>
            </div>
          ))}
          <button className="ghost wide" onClick={() => updateProject((draft) => {
            draft.obstacles.push({ x: 0, y: 1, radius: 0.2 });
          })}><Plus size={15} />添加障碍</button>

          <button className="primary wide" onClick={() => runSimulation(project, true)}>
            <Play size={16} />计算并播放完整运动
          </button>
        </aside>

        <section className="panel viewer">
          <Scene sample={currentSample} rejectionSample={rejectionSample}
            obstacles={project.obstacles} samples={samples} />
          <div className={"status-card " + (result?.accepted ? "accepted" : "rejected")}>
            <strong>{result?.accepted ? "轨迹通过" : "轨迹拒绝"}</strong>
            <span>总时间：{result?.duration.toFixed(2)} s</span>
            <span>{result?.rejection ? "碰撞/违规时间：" + result.rejection.time.toFixed(2) + " s" : "所有验证采样帧安全"}</span>
          </div>
          {message && <div className="message">{message}</div>}
        </section>

        <aside className="panel replay">
          <h2>运动重放</h2>
          <div className="replay-buttons">
            <button onClick={() => setPlaying(true)} disabled={!result?.accepted}><Play size={16} />播放</button>
            <button onClick={() => setPlaying(false)}><Pause size={16} />暂停</button>
            <button onClick={() => { setPlaying(false); setFrame(0); }}><RotateCcw size={16} />复位</button>
          </div>
          <input className="timeline" type="range" min={0} max={Math.max(0, samples.length - 1)}
            value={frame} onChange={(event) => { setPlaying(false); setFrame(Number(event.target.value)); }} />
          <div className="time-readout">{sliderTime.toFixed(2)} / {result?.duration.toFixed(2)} s</div>

          <h3>分段耗时（受速度上限约束）</h3>
          <ul className="duration-list">
            {result?.segment_durations.map((duration, index) => (
              <li key={index}><span>姿态 {index + 1} → {index + 2}</span><strong>{duration.toFixed(2)} s</strong></li>
            ))}
          </ul>

          <h3>当前关节角</h3>
          <div className="current-angles">
            {currentSample?.angles.map((angle, index) => (
              <div key={index}><span>J{index + 1}</span><strong>{angle.toFixed(1)}°</strong></div>
            ))}
          </div>

          {result?.rejection && (
            <div className="rejection-detail">
              <h3>拒绝原因</h3>
              <p>{result.rejection.message}</p>
              <p>问题帧角度：{result.rejection.angles.map((angle) => angle.toFixed(1) + "°").join("，")}</p>
              {typeof result.rejection.clearance === "number" && <p>穿透余量：{result.rejection.clearance.toFixed(3)} m</p>}
              <p className="hint">红色虚影是途中违规位置；起终点安全也不会放行。</p>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}
