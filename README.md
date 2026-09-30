# 机械臂离线编程平台

Python 负责机器人模型、逐采样轨迹、分段速度限速、衔接连续性核对和碰撞计算；React/Three.js 负责姿态编辑、分段速度编辑、抓放动作编辑、连杆/障碍物/携带物体展示、完整运动重放和工程保存。当前不连接真实机械臂，也不向控制器下发指令。

## 功能

- 三关节 2D 串联连杆，多个关节姿态之间做同步线性插值，检查完整途中位置而不是只查端点。
- 每段可独立设置各关节速度上限（`segment_speed_limits`），段耗时取该段角位移除以该段速度上限的最大值。
- 段交界处逐关节比对相邻段实际速度，差异超过 `junction_speed_tolerance` 时以 `speed_jump` 拒绝，避免突然的速度/方向突变。
- 每个采样帧检查关节限位、连杆与圆形障碍物、非相邻连杆自碰撞。
- 支持末端携带物体：物体静止时使用自身圆形外形，抓取后切换为跟随末端的物体外形（半径 + 持握偏移 `hold_dx/hold_dy`），运输途中同时检查物体与障碍、物体与机械臂连杆的碰撞。
- 抓/放动作绑定到途经姿态和物体名；抓取姿态与物体初始位置偏差超过 `grasp_tolerance` 时以 `grasp_unreachable` 拒绝。结果包含 `grasp_events`/`release_events` 的时刻。
- 每段独立标记 `segments_solved`；某段无解时前面已编辑的姿态仍保留，但整体不标记为可执行。
- 保存工程时在服务端重新核对整段轨迹，写入 `executable`、`constraint_version`、抓放时刻、所用物体和各段是否有解；重新加载即按同一约束版本独立重放核对。
- 起终点安全但中途碰撞时返回拒绝结果、首次碰撞时间和完整帧；前端在碰撞时刻标红并停止继续重放。
- 工程保存到后端 `projects/<id>.json`，支持列表加载、调整后更新。

## 启动

```bash
cd backend
PYTHONPATH=.. ../.venv/bin/python -m uvicorn robot_platform.main:app --reload --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

前端通过 Vite 的 `/api` 代理访问后端。

## 验证

```bash
PYTHONPATH=backend .venv/bin/python -m pytest backend/tests/test_robotics.py -q
cd frontend && npm run build
```

约束版本当前为 `grasp-multisegment-1`。默认工程在 P2 抓取 `block`、在 P3 放下，运输途中绿色方块为携带外形，静止时为紫色线框外形。早期的中间碰撞示例仍可通过把障碍放回扫掠路径复现：端点安全但第一连杆在途中穿过障碍，结果被拒绝；`test_robotics.py` 同时验证第一关节速度从 `1.0 rad/s` 降到 `0.5 rad/s` 时该段耗时从 `1.3s` 变为 `2.6s`。
