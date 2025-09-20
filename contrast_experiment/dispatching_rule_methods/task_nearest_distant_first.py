
import random
import math
import copy
import pickle
import os
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional

# =========================================================
# Configuration block
# =========================================================
ENV_CONFIG = {
    "env_paras": {
        "num_usvs": 5,
        "num_tasks": 24,
        "map_size": [120, 120],
        "battery_capacity": 220.0,
        "usv_speed": 5.0,
        "charge_time": 30.0,
        "min_task_time_visual": 5.0
    },
    "random_seed": 42,
    "task_service_time_range": (5.0, 20.0),
    "energy_cost_per_unit_distance": 1.0,
    "task_time_energy_ratio": 0.5,
    "usv_initial_position": (0.0, 0.0),
    "enable_task_random_priority": False
}
NUM_RUNS = 100
PROGRESS_BAR_WIDTH = 40

# === 修改：指定加载环境备份的路径 ===
ENV_BACKUP_FILE = r"E:\vsproject\usv-hgnn-ppo\contrast_experiment\dispatching_rule_methods\env_backup.pkl" # 指定绝对路径

# =========================================================
# Data Classes -- 新增序列化支持
# =========================================================
@dataclass
class Task:
    task_id: int
    position: Tuple[float, float]
    service_time: float
    priority: int = 0
    assigned_usv: Optional[int] = None
    start_time: Optional[float] = None
    finish_time: Optional[float] = None
    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

@dataclass
class USV:
    usv_id: int
    position: Tuple[float, float]
    battery_capacity: float
    battery_level: float
    speed: float
    charge_time: float
    timeline: List[Dict] = field(default_factory=list)
    current_time: float = 0.0
    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, data):
        return cls(**data)
    def distance_to(self, point: Tuple[float, float]) -> float:
        return math.sqrt((self.position[0] - point[0]) ** 2 + (self.position[1] - point[1]) ** 2)
    def can_execute(self, task: Task, env_params: Dict, energy_model) -> bool:
        travel_dist = self.distance_to(task.position)
        energy_need = energy_model(travel_dist, task.service_time)
        return self.battery_level >= energy_need
    def execute_task(self, task: Task, env_params: Dict, energy_model):
        travel_dist = self.distance_to(task.position)
        travel_time = travel_dist / self.speed
        energy_need = energy_model(travel_dist, task.service_time)
        start_time = self.current_time + travel_time
        finish_time = start_time + task.service_time
        self.current_time = finish_time
        self.battery_level -= energy_need
        self.position = task.position
        self.timeline.append({
            "type": "task",
            "task_id": task.task_id,
            "depart_time": start_time - travel_time,
            "arrive_time": start_time,
            "start_service": start_time,
            "finish_service": finish_time,
            "energy_used": energy_need,
            "battery_after": self.battery_level
        })
        task.assigned_usv = self.usv_id
        task.start_time = start_time
        task.finish_time = finish_time
    def charge_full(self):
        start_charge = self.current_time
        self.current_time += self.charge_time
        self.battery_level = self.battery_capacity
        self.timeline.append({
            "type": "charge",
            "start_charge": start_charge,
            "finish_charge": self.current_time,
            "battery_after": self.battery_level
        })

# =========================================================
# Environment -- 新增 save/load
# =========================================================
class Environment:
    def __init__(self, config: Dict, run_seed: Optional[int] = None):
        self.config = config
        self.params = config["env_paras"]
        if run_seed is not None:
            random.seed(run_seed)
        self.tasks: List[Task] = []
        self.usvs: List[USV] = []
        self._generate_tasks()
        self._generate_usvs()

    def _generate_tasks(self):
        num_tasks = self.params["num_tasks"]
        map_w, map_h = self.params["map_size"]
        st_min, st_max = ENV_CONFIG["task_service_time_range"]
        min_task_time_visual = self.params["min_task_time_visual"]
        for tid in range(num_tasks):
            x = random.uniform(0, map_w)
            y = random.uniform(0, map_h)
            service_time = random.uniform(st_min, st_max)
            service_time = max(service_time, min_task_time_visual)
            priority = 0
            if ENV_CONFIG["enable_task_random_priority"]:
                priority = random.randint(1, 5)
            self.tasks.append(Task(
                task_id=tid,
                position=(x, y),
                service_time=service_time,
                priority=priority
            ))

    def _generate_usvs(self):
        num_usvs = self.params["num_usvs"]
        init_pos = ENV_CONFIG["usv_initial_position"]
        for uid in range(num_usvs):
            self.usvs.append(
                USV(
                    usv_id=uid,
                    position=init_pos,
                    battery_capacity=self.params["battery_capacity"],
                    battery_level=self.params["battery_capacity"],
                    speed=self.params["usv_speed"],
                    charge_time=self.params["charge_time"]
                )
            )

    # —— 新增：保存/加载 —— #
    def save_environment(self, filename: str):
        env_data = {
            "tasks": [t.to_dict() for t in self.tasks],
            "usvs": [u.to_dict() for u in self.usvs],
            "config": self.config
        }
        with open(filename, 'wb') as f:
            pickle.dump(env_data, f)

    @classmethod
    def load_environment(cls, filename: str):
        with open(filename, 'rb') as f:
            env_data = pickle.load(f)
        # 先构造空环境（不触发随机），再回填数据
        env = cls(env_data["config"], run_seed=None)
        env.tasks = [Task.from_dict(t) for t in env_data["tasks"]]
        env.usvs = [USV.from_dict(u) for u in env_data["usvs"]]
        return env

# =========================================================
# Planner（修改任务排序和USV选择逻辑）
# =========================================================
class NearestTaskFirstPlanner: # 修改类名以反映策略
    def __init__(self, env: Environment):
        self.env = env
        self.params = env.params
        self.energy_cost_per_unit_distance = ENV_CONFIG["energy_cost_per_unit_distance"]
        self.task_time_energy_ratio = ENV_CONFIG["task_time_energy_ratio"]

    def energy_model(self, distance: float, service_time: float) -> float:
        return distance * self.energy_cost_per_unit_distance + service_time * self.task_time_energy_ratio

    def _task_distance_from_base(self, task: Task) -> float:
        bx, by = ENV_CONFIG["usv_initial_position"]
        return math.sqrt((task.position[0] - bx) ** 2 + (task.position[1] - by) ** 2)

    def plan(self):
        tasks = self.env.tasks[:]
        # === 修改：按距离基地从近到远排序 (近任务优先) ===
        tasks.sort(key=lambda t: self._task_distance_from_base(t))

        for task in tasks:
            # === 修改：USV选择逻辑 ===
            # 按距离任务远近排序，距离近的优先
            candidates = []
            for usv in self.env.usvs:
                dist = usv.distance_to(task.position)
                candidates.append((dist, usv.usv_id, usv))
            candidates.sort(key=lambda x: (x[0], x[1])) # 按距离，然后按ID排序

            assigned = False
            for _, _, usv in candidates:
                if usv.can_execute(task, self.params, self.energy_model):
                    usv.execute_task(task, self.params, self.energy_model)
                    assigned = True
                    break
                else:
                    usv.charge_full()
                    if usv.can_execute(task, self.params, self.energy_model):
                        usv.execute_task(task, self.params, self.energy_model)
                        assigned = True
                        break
            # 注意：这里移除了原文件中的 if not assigned: pass，因为未分配的任务不需要特殊处理

# =========================================================
# Progress Bar (保持不变)
# =========================================================
def print_progress(current: int, total: int):
    ratio = (current + 1) / total
    filled = int(ratio * PROGRESS_BAR_WIDTH)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    print(f"\rProgress [{bar}] {current + 1}/{total} ({ratio * 100:5.1f}%)", end="", flush=True)

# =========================================================
# Single Run (修改以使用加载的环境)
# =========================================================
# === 修改：run_one 函数签名和逻辑 ===
# 移除 run_index 和 run_seed，改为接收一个 Environment 对象
def run_one(env: Environment) -> Dict:
    # env = Environment(ENV_CONFIG, run_seed=run_seed) # 不再需要，环境已传入
    planner = NearestTaskFirstPlanner(env) # 使用新的 Planner
    planner.plan()
    tasks = env.tasks
    total_tasks = len(tasks)
    assigned_tasks = [t for t in tasks if t.assigned_usv is not None]
    assigned_count = len(assigned_tasks)
    
    # === 修改：总是计算完成时间 ===
    completion_time = float('inf') # 默认为无穷大
    if assigned_count > 0:
        completion_time = max(t.finish_time for t in assigned_tasks if t.finish_time is not None)
        # 如果 max 返回 None (理论上不应该)，completion_time 保持 inf
    
    all_finished = (assigned_count == total_tasks)
    
    # === 修改：snapshot 结构 ===
    # 为了与原文件保持一致，这里仍然返回一个包含 tasks 和 usvs 深拷贝的字典
    snapshot = {
        # "run_index": run_index, # run_index 不再传入
        "all_finished": all_finished,
        "assigned_count": assigned_count,
        "total_tasks": total_tasks,
        "completion_time": completion_time, # 总是包含完成时间
        "tasks": copy.deepcopy(env.tasks),
        "usvs": copy.deepcopy(env.usvs)
    }
    return snapshot

# =========================================================
# Best Selection Logic (保持不变)
# =========================================================
def better(a: Dict, b: Optional[Dict]) -> bool:
    if b is None:
        return True
    if a["all_finished"] and not b["all_finished"]:
        return True
    if b["all_finished"] and not a["all_finished"]:
        return False
    if a["all_finished"] and b["all_finished"]:
        return a["completion_time"] < b["completion_time"]
    if a["assigned_count"] != b["assigned_count"]:
        return a["assigned_count"] > b["assigned_count"]
    # 如果完成任务数也相同，则比较完成时间（即使是 partial run）
    return a["completion_time"] < b["completion_time"]

# =========================================================
# Output Best Result (修改完成时间输出)
# =========================================================
def print_best_result(result: Dict, planner_label: str = "Near Tasks First + On-Demand Charge"):
    print("\n================ BEST RESULT ================")
    print(f"Strategy: {planner_label}")
    # print(f"Best Run: {result['run_index'] + 1}/{NUM_RUNS}") # run_index 不再使用
    print(f"Total Tasks: {result['total_tasks']}")
    print(f"Completed: {result['assigned_count']}")
    
    # === 修改：总是输出完成时间 ===
    completion_time_output = "N/A"
    if result['assigned_count'] > 0 and result['completion_time'] < float('inf'):
        completion_time_output = f"{result['completion_time']:.2f}"

    if result["all_finished"]:
        print("All Tasks Completed: YES")
        print(f"Total completion time (all tasks finished): {completion_time_output}")
    else:
        print("All Tasks Completed: NO (showing best partial run)")
        if completion_time_output != "N/A":
             print(f"Total completion time (partial run): {completion_time_output}")
        else:
             print("Total completion time: N/A")
        
    print("\n--- Task Assignment Details ---")
    tasks_sorted = sorted(result["tasks"], key=lambda t: t.task_id)
    bx, by = ENV_CONFIG["usv_initial_position"]
    def dist_base(t):
        return math.sqrt((t.position[0] - bx) ** 2 + (t.position[1] - by) ** 2)
    for t in tasks_sorted:
        if t.assigned_usv is not None:
            print(f"Task {t.task_id:02d} -> USV {t.assigned_usv} "
                  f"Start={t.start_time:.2f} Finish={t.finish_time:.2f} "
                  f"DistBase={dist_base(t):.1f} Pos=({t.position[0]:.1f},{t.position[1]:.1f}) "
                  f"Service={t.service_time:.1f}")
        else:
            print(f"Task {t.task_id:02d} -> Unassigned DistBase={dist_base(t):.1f} "
                  f"Pos=({t.position[0]:.1f},{t.position[1]:.1f})")
    print("\n--- USV Timelines ---")
    for usv in result["usvs"]:
        print(f"\nUSV {usv.usv_id} FinalTime={usv.current_time:.2f} BatteryRemaining={usv.battery_level:.2f}")
        for entry in usv.timeline:
            if entry["type"] == "charge":
                print(f"  [Charge] {entry['start_charge']:.2f}->{entry['finish_charge']:.2f} "
                      f"Battery={entry['battery_after']:.2f}")
            elif entry["type"] == "task":
                print(f"  [Task {entry['task_id']}] Depart={entry['depart_time']:.2f} "
                      f"Arrive={entry['arrive_time']:.2f} "
                      f"SvcStart={entry['start_service']:.2f} SvcFinish={entry['finish_service']:.2f} "
                      f"EnergyUsed={entry['energy_used']:.2f} BatteryAfter={entry['battery_after']:.2f}")
    
    # === 新增：在 END 之前再次输出 Total completion time (all tasks finished): ===
    print(f"\nTotal completion time (all tasks finished): {completion_time_output}") 
    
    print("\n================ END =================\n")

# =========================================================
# Multi-run orchestration (修改加载逻辑和主循环)
# =========================================================
def run_multiple(num_runs: int = NUM_RUNS):
    """
    强制从指定路径加载已固定的环境进行多轮运行。
    """
    # 1. 决定环境列表 envs
    # === 修改：强制从指定路径加载 ===
    if os.path.exists(ENV_BACKUP_FILE):
        print(f"【加载】从指定路径读取已固定环境: {ENV_BACKUP_FILE}")
        # === 修改：直接加载 Environment 对象列表 ===
        # 假设 env_backup.pkl 保存的是 [env_obj1, env_obj2, ...] (由 usv_task_random_planner.py 生成)
        with open(ENV_BACKUP_FILE, 'rb') as f:
            envs = pickle.load(f)
        # --- 检查加载的确实是 Environment 对象 ---
        if not envs or not all(isinstance(env, Environment) for env in envs):
             raise TypeError("Loaded data is not a list of Environment objects.")

        if len(envs) != num_runs:
            print(f"警告：备份文件中的环境数量 ({len(envs)}) 与当前 NUM_RUNS ({num_runs}) 不符！")
            # 可以选择截断或报错
            # 这里假设文件是正确的，使用文件中的数量
            num_runs = len(envs)
            print(f"将运行 {num_runs} 轮。")
    else:
        # === 修改：如果指定路径文件不存在，则报错 ===
        raise FileNotFoundError(f"指定的环境备份文件不存在: {ENV_BACKUP_FILE}. 请确保文件已由 usv_task_random_planner.py 生成。")

    # 2. 依次规划
    print(f"Running {num_runs} iterations using Nearest Task First strategy...") # 修改提示信息
    best_result = None
    # === 修改：主循环逻辑 ===
    # 移除 run_index 循环，改为遍历 envs 列表
    for i, env in enumerate(envs):
        # result = run_one(run_idx) # 原逻辑
        result = run_one(env) # 修改：传入已加载的环境对象
        if better(result, best_result):
            best_result = result
        # print_progress(run_idx, NUM_RUNS) # 原逻辑
        print_progress(i, num_runs) # 修改：使用索引 i
    print()  # newline after progress bar
    print_best_result(best_result) # 保持不变

# =========================================================
# Main (修改调用逻辑)
# =========================================================
def main():
    # === 修改：默认加载固定环境 ===
    try:
        run_multiple(NUM_RUNS)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保环境备份文件已存在于指定路径。")
    except Exception as e:
        print(f"运行时发生错误: {e}")
        raise

if __name__ == "__main__":
    main()