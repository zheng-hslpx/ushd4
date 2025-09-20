"""
usv_task_random_planner.py  支持“第一次随机后固定环境”版本
方案一：保存 & 加载环境数据（pickle）
用法见文件末尾 “==== 使用步骤 ====”
"""
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
    "random_seed": 42,            # 基础种子
    "task_service_time_range": (5.0, 20.0),
    "energy_cost_per_unit_distance": 1.0,
    "task_time_energy_ratio": 0.5,
    "usv_initial_position": (0.0, 0.0),
    "enable_task_random_priority": False
}

NUM_RUNS = 100                  # 想跑多少轮
ENV_BACKUP_FILE = "env_backup.pkl"  # 保存第一次随机出来的 100 组环境
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
#  Planner / 多轮运行 / main 保持原逻辑不变
# =========================================================
class SimpleRandomPlanner:
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

    def plan(self):
        tasks = self.env.tasks[:]
        random.shuffle(tasks)
        for task in tasks:
            candidate_usvs = self.env.usvs[:]
            random.shuffle(candidate_usvs)
            assigned = False
            for usv in candidate_usvs:
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
                    if not usv.can_execute(task, self.params, self.energy_model):
                        self.warnings.append(
                            f"[Warning] Task {task.task_id} still infeasible for USV {usv.usv_id} even after full charge."
                        )
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
        print("\n====== BEST RUN ASSIGNMENT SUMMARY ======")
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


def run_multiple(num_runs: int = NUM_RUNS, *, reload: bool = False):
    """
    reload=False -> 第一次生成并保存
    reload=True  -> 直接读备份，不再随机
    """
    base_seed = ENV_CONFIG.get("random_seed", None)
    best_result = None
    best_env = None
    best_planner = None
    best_run_index = -1

    # 1. 决定环境列表 envs
    if reload and os.path.exists(ENV_BACKUP_FILE):
        print("【加载】读取已固定环境 …")
        with open(ENV_BACKUP_FILE, 'rb') as f:
            envs = pickle.load(f)
        assert len(envs) == num_runs, "备份文件轮次数与当前 NUM_RUNS 不符，请检查！"
    else:
        print("【生成】第一次随机环境 …")
        envs = []
        for i in range(num_runs):
            run_seed = (base_seed + i) if base_seed is not None else None
            envs.append(Environment(ENV_CONFIG, run_seed=run_seed))
        # 保存
        with open(ENV_BACKUP_FILE, 'wb') as f:
            pickle.dump(envs, f)
        print(f"【保存】{num_runs} 组环境已写入 {ENV_BACKUP_FILE}")

    # 2. 依次规划
    print(f"Running {num_runs} iterations...")
    for i, env in enumerate(envs):
        planner = SimpleRandomPlanner(env, log_enabled=False)
        planner.plan()
        metrics = planner.compute_metrics()

        better = False
        if best_result is None:
            better = True
        else:
            if metrics["all_assigned"] and not best_result["all_assigned"]:
                better = True
            elif metrics["all_assigned"] == best_result["all_assigned"]:
                if metrics["total_assigned"] > best_result["total_assigned"]:
                    better = True
                elif metrics["total_assigned"] == best_result["total_assigned"]:
                    if metrics["max_finish_time"] is not None and best_result["max_finish_time"] is not None:
                        if metrics["max_finish_time"] < best_result["max_finish_time"]:
                            better = True
        if better:
            best_result, best_env, best_planner, best_run_index = metrics, env, planner, i + 1

        # 进度条
        progress = (i + 1) / num_runs
        bar_len = 40
        filled = int(bar_len * progress)
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"\rProgress: [{bar}] {i+1}/{num_runs} ({progress*100:5.1f}%)", end="", flush=True)

    print()
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


def main():
    # 第一次跑：main()
    # 以后固定数据：main(reload=True)
    import sys
    reload = "--reload" in sys.argv
    run_multiple(NUM_RUNS, reload=reload)


if __name__ == "__main__":
    main()