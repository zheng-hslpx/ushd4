
import random
import math
import copy
import pickle
import os
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional

# =========================================================
# 配置区
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
    "random_seed": 42,            # 基础种子 (环境固定后此种子对环境生成无效)
    "task_service_time_range": (5.0, 20.0),
    "energy_cost_per_unit_distance": 1.0,
    "task_time_energy_ratio": 0.5,
    "usv_initial_position": (0.0, 0.0),
    "enable_task_random_priority": False
}
NUM_RUNS = 100                  # 想跑多少轮

# === 修改：指定加载环境备份的路径 ===
ENV_BACKUP_FILE = r"E:\vsproject\usv-hgnn-ppo\contrast_experiment\dispatching_rule_methods\env_backup.pkl" # 指定绝对路径

# =========================================================
# 数据类 –– 新增序列化支持
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
        # Handle potential missing keys if needed, but assuming data is complete
        return cls(**data)
    # —— 以下方法保持原样 —— #
    def distance_to(self, point: Tuple[float, float]) -> float:
        return math.sqrt((self.position[0] - point[0])**2 + (self.position[1] - point[1])**2)
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
# 环境 –– 新增 save/load
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
            service_time = max(random.uniform(st_min, st_max), min_task_time_visual)
            priority = random.randint(1, 5) if ENV_CONFIG["enable_task_random_priority"] else 0
            self.tasks.append(Task(
                task_id=tid, position=(x, y), service_time=service_time, priority=priority
            ))

    def _generate_usvs(self):
        num_usvs = self.params["num_usvs"]
        init_pos = ENV_CONFIG["usv_initial_position"]
        for uid in range(num_usvs):
            self.usvs.append(
                USV(usv_id=uid, position=init_pos,
                    battery_capacity=self.params["battery_capacity"],
                    battery_level=self.params["battery_capacity"],
                    speed=self.params["usv_speed"],
                    charge_time=self.params["charge_time"])
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
# Planner / 多轮运行 / main (修改规划逻辑和加载路径处理)
# =========================================================

# === 修改：Planner 类名以反映新策略 ===
class HighestBatteryFirstPlanner: # 原名 SimpleRandomPlanner
    def __init__(self, env: Environment, log_enabled: bool = True):
        self.env = env
        self.params = env.params
        self.energy_cost_per_unit_distance = ENV_CONFIG["energy_cost_per_unit_distance"]
        self.task_time_energy_ratio = ENV_CONFIG["task_time_energy_ratio"]
        self.log_enabled = log_enabled
        self.warnings: List[str] = []
        self.failures: List[str] = []

    def _log(self, msg: str):
        if self.log_enabled:
            print(msg)

    def energy_model(self, distance: float, service_time: float) -> float:
        return distance * self.energy_cost_per_unit_distance + service_time * self.task_time_energy_ratio

    # === 修改：plan 方法实现 Highest Battery First 策略 ===
    def plan(self):
        # 任务可以随机打乱顺序 (可选，保持一定随机性)
        tasks = self.env.tasks[:]
        random.shuffle(tasks)

        for task in tasks:
            # === 核心修改：按当前电量从高到低排序 USV ===
            # 如果电量相同，则按 USV ID 排序以保证确定性
            # 使用 -u.battery_level 实现降序排列
            candidate_usvs = sorted(
                self.env.usvs,
                key=lambda u: (-u.battery_level, u.usv_id)
            )

            assigned = False
            for usv in candidate_usvs:
                if usv.can_execute(task, self.params, self.energy_model):
                    usv.execute_task(task, self.params, self.energy_model)
                    assigned = True
                    break
                else:
                    # 尝试充电后再执行
                    usv.charge_full()
                    if usv.can_execute(task, self.params, self.energy_model):
                        usv.execute_task(task, self.params, self.energy_model)
                        assigned = True
                        break
                    # 如果充电后仍然无法执行，记录警告（但继续尝试下一个USV）

            # 如果所有USV都无法分配该任务，则记录失败
            if not assigned:
                 self.failures.append(f"[Failure] Task {task.task_id} could not be assigned.")

    def compute_metrics(self) -> Dict:
        total_assigned = sum(1 for t in self.env.tasks if t.assigned_usv is not None)
        all_assigned = (total_assigned == len(self.env.tasks))
        max_finish_time = None
        if total_assigned > 0:
            max_finish_time = max(t.finish_time for t in self.env.tasks if t.finish_time is not None)
        return {
            "total_assigned": total_assigned,
            "all_assigned": all_assigned,
            "max_finish_time": max_finish_time
        }

    def summary(self):
        print("\n====== BEST RUN ASSIGNMENT SUMMARY (Highest Battery First) ======") # 修改摘要标题
        total_assigned = 0
        for task in sorted(self.env.tasks, key=lambda t: t.task_id):
            if task.assigned_usv is not None:
                total_assigned += 1
                print(f"Task {task.task_id:02d} -> USV {task.assigned_usv}, "
                      f"Start={task.start_time:.2f}, Finish={task.finish_time:.2f}, "
                      f"Pos={task.position}, ServiceTime={task.service_time:.1f}")
            else:
                print(f"Task {task.task_id:02d} -> Unassigned")
        print(f"Assigned Tasks: {total_assigned}/{len(self.env.tasks)}")
        for usv in self.env.usvs:
            print(f"\n--- USV {usv.usv_id} Timeline (FinalTime={usv.current_time:.2f}, BatteryRemaining={usv.battery_level:.2f}) ---")
            for entry in usv.timeline:
                if entry["type"] == "charge":
                    print(f"  [Charge] {entry['start_charge']:.2f} -> {entry['finish_charge']:.2f} Battery={entry['battery_after']:.2f}")
                elif entry["type"] == "task":
                    print(f"  [Task {entry['task_id']}] Depart={entry['depart_time']:.2f} Arrive={entry['arrive_time']:.2f} "
                          f"StartService={entry['start_service']:.2f} FinishService={entry['finish_service']:.2f} "
                          f"EnergyUsed={entry['energy_used']:.2f} Battery={entry['battery_after']:.2f}")
        if total_assigned > 0:
            max_finish_time = max(t.finish_time for t in self.env.tasks if t.finish_time is not None)
            if total_assigned == len(self.env.tasks):
                print(f"\nTotal completion time (all tasks finished): {max_finish_time:.2f}")
            else:
                print(f"\nPartial completion time (not all tasks assigned): {max_finish_time:.2f}")
        else:
            print("\nNo tasks were completed, total completion time unavailable.")


def run_multiple(num_runs: int = NUM_RUNS):
    """
    强制从指定路径加载已固定的环境进行多轮运行。
    """
    best_result = None
    best_env = None
    best_planner = None
    best_run_index = -1

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
    print(f"Running {num_runs} iterations using Highest Battery First strategy...") # 修改提示信息
    for i, env in enumerate(envs):
        # === 修改：使用新的 HighestBatteryFirstPlanner ===
        planner = HighestBatteryFirstPlanner(env, log_enabled=False)
        planner.plan()
        metrics = planner.compute_metrics()
        better = False
        if best_result is None:
            better = True
        else:
            # 优化目标：优先完成所有任务，其次完成任务数多，最后完成时间短
            if metrics["all_assigned"] and not best_result["all_assigned"]:
                better = True
            elif metrics["all_assigned"] == best_result["all_assigned"]:
                if metrics["total_assigned"] > best_result["total_assigned"]:
                    better = True
                elif metrics["total_assigned"] == best_result["total_assigned"]:
                    if metrics["max_finish_time"] is not None and best_result["max_finish_time"] is not None:
                        if metrics["max_finish_time"] < best_result["max_finish_time"]:
                            better = True
                    # 如果当前结果时间是None(没完成任务)，而best不是None，则不替换
                    elif metrics["max_finish_time"] is not None and best_result["max_finish_time"] is None:
                         better = True # 当前有时间，best没时间，当前更好？不，应该比较完成任务数。上面逻辑已覆盖。

        if better:
            best_result, best_env, best_planner, best_run_index = metrics, env, planner, i + 1

        # 进度条
        progress = (i + 1) / num_runs
        bar_len = 40
        filled = int(bar_len * progress)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\rProgress: [{bar}] {i+1}/{num_runs} ({progress*100:5.1f}%)", end="", flush=True)
    print() # 换行

    print("========================================")
    print(f"Best run index: {best_run_index}/{num_runs}")
    if best_result:
        print(f"All tasks assigned: {best_result['all_assigned']}")
        print(f"Assigned count: {best_result['total_assigned']}")
        if best_result['max_finish_time'] is not None:
            print(f"Best total completion time: {best_result['max_finish_time']:.2f}")
    print("========================================")
    if best_planner is not None:
        best_planner.summary()
    else:
        print("No valid runs were produced.")

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