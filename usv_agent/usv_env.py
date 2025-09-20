from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
from gymnasium import spaces


@dataclass
class USVState:
    id: int
    position: np.ndarray
    battery: float
    status: str
    current_task: Optional[int] = None
    available_time: float = 0.0
    total_distance: float = 0.0
    work_time: float = 0.0
    assigned_tasks: Optional[List[int]] = None

    def __post_init__(self):
        if self.assigned_tasks is None:
            self.assigned_tasks = []


@dataclass
class TaskState:
    id: int
    position: np.ndarray
    processing_time: float
    fuzzy_time: Tuple[float, float, float]
    status: str
    assigned_usv: Optional[int] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None


class USVEnv:
    """USV 任务调度环境（统一分配版）"""

    def __init__(self, env_config: Dict[str, Any]):
        self.num_usvs: int = int(env_config['num_usvs'])
        self.num_tasks: int = int(env_config['num_tasks'])
        self.map_size: np.ndarray = np.array(env_config['map_size'], dtype=np.float32)
        self.usv_speed: float = float(env_config.get('usv_speed', 5.0))

        self.action_space = spaces.Discrete(self.num_usvs * self.num_tasks)

        self.usvs: List[USVState] = []
        self.tasks: List[TaskState] = []
        self.schedule_history: List[Dict[str, float]] = []
        self.makespan: float = 0.0
        self.done: bool = False
        self.step_count: int = 0
        self.debug_mode: bool = False
        self.pending_assignments: List[Tuple[int, int]] = []

    def reset(self, tasks_data: Optional[List[TaskState]] = None,
              usvs_data: Optional[List[USVState]] = None) -> Dict[str, np.ndarray]:
        self.makespan, self.done, self.step_count = 0.0, False, 0
        self.schedule_history.clear()
        self.pending_assignments.clear()

        self.usvs = usvs_data or [
            USVState(
                id=i,
                position=np.zeros(2, dtype=np.float32),
                battery=float('inf'),
                status='idle'
            ) for i in range(self.num_usvs)
        ]

        self.tasks = tasks_data or [
            TaskState(
                id=i,
                position=np.random.uniform(0, self.map_size, 2).astype(np.float32),
                processing_time=float(np.random.uniform(8.0, 30.0)),
                fuzzy_time=(0.0, 0.0, 0.0),
                status='unscheduled'
            ) for i in range(self.num_tasks)
        ]

        self._update_current_makespan()
        return self._get_observation()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        usv_feats = np.array([[*u.position, u.available_time] for u in self.usvs], dtype=np.float32)
        task_feats = np.array([[*t.position, t.processing_time, 1 if t.status == 'unscheduled' else 0]
                               for t in self.tasks], dtype=np.float32)
        return {
            "usv_features": usv_feats,
            "task_features": task_feats,
            "action_mask": self._compute_action_mask()
        }

    def _compute_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_usvs * self.num_tasks, dtype=np.int8)
        if self.num_usvs == 0 or self.num_tasks == 0:
            return mask

        # 找到所有空闲 USV（available_time <= 当前 makespan）
        idle_usvs = [ui for ui, u in enumerate(self.usvs) if u.available_time <= self.makespan + 1e-6]
        for ui in idle_usvs:
            for ti, t in enumerate(self.tasks):
                if t.status == 'unscheduled':
                    mask[ui * self.num_tasks + ti] = 1
        return mask

    def step(self, action: int):
        if all(t.status != "unscheduled" for t in self.tasks):
            self.done = True
            return self._get_observation(), 0.0, True, {"makespan": float(self.makespan), "step_count": int(self.step_count)}

        # 解析动作
        usv_idx = action // self.num_tasks
        task_idx = action % self.num_tasks

        prev_ms = float(self.makespan)

        # 添加到待分配列表
        self.pending_assignments.append((usv_idx, task_idx))

        # 关键修改：**每步都尝试执行一次统一分配**
        # 但只在有任务可分配且存在空闲 USV 时才执行
        if len(self.pending_assignments) > 0 and any(u.available_time <= self.makespan + 1e-6 for u in self.usvs):
            self._execute_pending_assignments()
            self._update_current_makespan()

        reward = float(prev_ms - self.makespan)
        self.step_count += 1
        self.done = all(t.status != "unscheduled" for t in self.tasks)

        if self.debug_mode and (self.step_count % 5 == 0 or self.done):
            print(f"[DEBUG] step={self.step_count:03d} ms={self.makespan:.2f}")

        info = {"makespan": float(self.makespan), "step_count": int(self.step_count)}
        return self._get_observation(), reward, self.done, info

    def _execute_pending_assignments(self) -> None:
        """统一分配所有待分配任务"""
        if not self.pending_assignments:
            return

        # 按 USV 分组
        usv_task_map = {}
        for usv_idx, task_idx in self.pending_assignments:
            if usv_idx not in usv_task_map:
                usv_task_map[usv_idx] = []
            usv_task_map[usv_idx].append(task_idx)

        # 执行分配
        for usv_idx, task_indices in usv_task_map.items():
            for task_idx in task_indices:
                self._assign_task_to_usv(usv_idx, task_idx)

        # 清空
        self.pending_assignments.clear()

    def _assign_task_to_usv(self, usv_idx: int, task_idx: int) -> None:
        u, t = self.usvs[usv_idx], self.tasks[task_idx]
        if t.status != 'unscheduled':
            return

        decision_time = u.available_time
        travel_distance = float(np.linalg.norm(u.position - t.position))
        travel_time = travel_distance / max(self.usv_speed, 1e-6)
        start_time = decision_time + travel_time
        completion_time = start_time + float(t.processing_time)

        # 更新 USV
        u.position = t.position.copy()
        u.available_time = completion_time
        u.status = 'working'
        u.assigned_tasks.append(task_idx)

        # 更新 Task
        t.status = 'scheduled'
        t.assigned_usv = usv_idx
        t.start_time = start_time
        t.completion_time = completion_time

        # 记录历史
        self.schedule_history.append({
            "usv": usv_idx,
            "task": task_idx,
            "start_time": float(start_time),
            "completion_time": float(completion_time),
            "travel_time": float(travel_time),
            "travel_distance": float(travel_distance),
        })

    def _update_current_makespan(self) -> None:
        if not self.usvs:
            self.makespan = 0.0
            return
        self.makespan = max(u.available_time for u in self.usvs)

    def get_balance_metrics(self) -> Dict[str, float]:
        cnts = [len(u.assigned_tasks) for u in self.usvs]
        sum_sq = sum(x * x for x in cnts) if cnts else 0.0
        jain = (sum(cnts) ** 2) / (self.num_usvs * sum_sq) if sum_sq > 0 else 1.0
        var = float(np.var(cnts) if cnts else 0.0)
        return {"jains_index": float(jain), "task_load_variance": var}

    def set_debug_mode(self, enabled: bool) -> None:
        self.debug_mode = bool(enabled)