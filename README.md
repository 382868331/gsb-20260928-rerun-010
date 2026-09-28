# 机械臂离线编程平台

Python 负责机器人模型、逐采样轨迹、速度限速和碰撞计算；React/Three.js 负责姿态编辑、连杆与障碍物展示、完整运动重放和工程保存。当前不连接真实机械臂，也不向控制器下发指令。

## 功能

- 三关节 2D 串联连杆，可继续在同一数据模型上扩展抓取动作。
- 多个关节姿态之间做同步线性插值，检查完整途中位置而不是只查端点。
- 每段耗时取各关节角位移除以速度上限的最大值，速度上限直接影响运动时间。
- 每个采样帧检查关节限位、连杆与圆形障碍物、非相邻连杆自碰撞。
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

默认工程从 `-0.65,0,0` 运动到 `0.65,0,0`。两个端点不接触位于扫掠路径中的 `(1.75,0.12)` 圆形障碍，但第一连杆会在途中穿过障碍，因此结果被拒绝。测试同时验证第一关节速度从 `1.0 rad/s` 降到 `0.5 rad/s` 时，该段耗时从 `1.3s` 变为 `2.6s`。
