# 机械臂离线编程平台

三自由度平面机械臂的离线关节轨迹编辑、碰撞检查与回放原型。Python/FastAPI 负责机器人运动学、速度限幅、关节限位、自碰撞和障碍碰撞；React + Three.js 负责编辑、保存和可视化，不连接真实机械臂。

## 启动

- 后端：`.venv/bin/uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`
- 前端：在 `frontend` 目录执行 `npm run dev`
- 工程 JSON 保存在 `data/projects/`

## 验证

- 后端：`.venv/bin/python -m pytest`
- 前端：在 `frontend` 目录执行 `npm run build`

默认工程的起终点均不碰 `(1.8, 0)` 障碍，但 J1 从 -70° 线性运动到 70° 时第二根连杆在约 0.87 秒穿过障碍，因此后端返回 `accepted=false` 和途中碰撞时间。
