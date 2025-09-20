# -*- coding: utf-8 -*-
# 9.11修改_总览：评估改为“固定算例优先”，并与训练保持一致的边特征/确定性策略
import json, argparse, numpy as np, torch
from pathlib import Path

from usv_agent.usv_env import USVEnv
from usv_agent.hgnn_model import HeterogeneousGNN
# 9.11修改_序号1：使用当前工程的 EnhancedPPOAgent
from usv_agent.ppo_policy import EnhancedPPOAgent

# 9.11修改_序号2：引入固定算例生成器（仅评估用，不影响训练随机性）
from usv_agent.data_generator import USVTaskDataGenerator

torch.set_default_dtype(torch.float32)

# ------------ 公共函数 ------------
def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def pick_device(model_cfg: dict) -> torch.device:
    want = str(model_cfg.get("device", "auto")).lower()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if want.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，改用 CPU")
        return torch.device("cpu")
    return torch.device(want)

# 9.11修改_序号3：与 train.py 一致的“轻量边特征”
@torch.no_grad()
def compute_lookahead_edges(state: dict, map_size, device: torch.device) -> torch.Tensor:
    """
    输入 state：{'usv_features':[U,3], 'task_features':[T,4]}
    返回张量 [U,T,3]：
      0) dist_norm      USV-Task 欧氏距离 / 地图对角线
      1) task_proximity 任务之间的最小邻距（仅未调度任务），广播到 U
      2) usv_opp        USV 到最近未调度任务的距离，广播到 T
    """
    uf = torch.as_tensor(state["usv_features"], dtype=torch.float32, device=device)   # [U,3]
    tf = torch.as_tensor(state["task_features"], dtype=torch.float32, device=device)  # [T,4]
    U, T = uf.size(0), tf.size(0)
    if U == 0 or T == 0:
        return torch.zeros((U, T, 3), dtype=torch.float32, device=device)

    usv_pos = uf[:, :2]                   # [U,2]
    task_pos = tf[:, :2]                  # [T,2]
    active   = tf[:, 3] > 0               # [T] 1=未完成

    # U×T 距离
    dist_ut = torch.cdist(usv_pos, task_pos)  # [U,T]

    # 任务最小邻距（仅 active）
    prox = torch.zeros(T, device=device)
    if active.sum() > 1:
        pos_a = task_pos[active]
        d_tt = torch.cdist(pos_a, pos_a)
        d_tt.fill_diagonal_(float("inf"))
        min_d, _ = torch.min(d_tt, dim=1)
        prox[active] = min_d
    feat_task_prox = prox.unsqueeze(0).expand(U, -1)   # [U,T]

    # USV 到最近 active 任务的距离，广播到 T
    if active.any():
        d_ua = torch.cdist(usv_pos, task_pos[active])  # [U, A]
        min_du, _ = torch.min(d_ua, dim=1)             # [U]
        feat_usv_opp = min_du.unsqueeze(1).expand(-1, T)  # [U,T]
    else:
        feat_usv_opp = torch.zeros(U, T, device=device)

    # 对角线归一
    diag = torch.norm(torch.as_tensor(map_size, dtype=torch.float32, device=device))
    if diag > 0:
        dist_ut        = dist_ut / diag
        feat_task_prox = feat_task_prox / diag
        feat_usv_opp   = feat_usv_opp / diag

    return torch.stack([dist_ut, feat_task_prox, feat_usv_opp], dim=-1)  # [U,T,3]

# 9.11修改_序号4：根据配置构建“固定评估算例”（由 seeds 决定，可复现）
def _build_fixed_eval_cases_from_cfg(cfg: dict):
    """
    返回：[(usvs, tasks), ...]；当 eval_fixed_cases.enabled=false 或无 seeds 时返回空列表，表示回退随机评估
    """
    e = (cfg.get("eval_fixed_cases") or {})
    if not e.get("enabled", False):
        return []

    envp = cfg.get("env_paras", {})
    seeds = list(e.get("seeds", []))
    if not seeds:
        return []

    gen_cfg = {
        "num_usvs": int(envp.get("num_usvs", 5)),
        "num_tasks": int(envp.get("num_tasks", 24)),
        "map_size": envp.get("map_size", [120, 120]),
        "battery_capacity": float(envp.get("battery_capacity", 220.0)),
        # （可选）评估时单独覆盖处理时间范围/分布；未提供则沿用环境默认
        "min_processing_time": float(e.get("min_processing_time", envp.get("min_processing_time", 8.0))),
        "max_processing_time": float(e.get("max_processing_time", envp.get("max_processing_time", 30.0))),
        "task_distribution": e.get("task_distribution", envp.get("task_distribution", "uniform")),
    }
    gen = USVTaskDataGenerator(gen_cfg)
    return [gen.generate_instance(seed=s) for s in seeds]

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path("config") / "improved_config.json"))
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent .pt（建议用训练得到的 best_makespan.pt 或某个快照）")
    # 9.11修改_序号5：未启用固定评估时，仍可用原先的随机评估回合数
    parser.add_argument("--random_fallback_eps", type=int, default=5, help="未启用固定评估时的随机评估回合数")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = USVEnv(cfg['env_paras'])
    model = HeterogeneousGNN(cfg['model_paras'])
    device = pick_device(cfg.get("model_paras", {}))

    agent = EnhancedPPOAgent(model, cfg['model_paras']).to(device)
    agent.set_train_mode(False)

    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"[Eval] Loaded checkpoint: {args.checkpoint}")

    # 优先使用“固定算例评估”；若未开启或无 seeds，则回退到“随机评估”
    cases = _build_fixed_eval_cases_from_cfg(cfg)

    makespans, rewards = [], []
    if cases:
        print(f"[Eval] Using FIXED deterministic cases: {len(cases)} seeds")
        for (usvs, tasks) in cases:
            s = env.reset(tasks_data=tasks, usvs_data=usvs)  # ★ 固定实例（关键改动）
            done, ep_rew, steps = False, 0.0, 0
            max_steps = env.num_usvs * env.num_tasks + 5
            info = {}
            while not done and steps < max_steps:
                edges = compute_lookahead_edges(s, env.map_size, device=device)
                a, _, _ = agent.get_action(s, edges, epoch=0, max_epoch=1, deterministic=True)
                s, r, done, info = env.step(a)
                ep_rew += float(r)
                steps += 1
            makespans.append(info.get('makespan', 0.0))
            rewards.append(ep_rew)
    else:
        print(f"[Eval] Using RANDOM deterministic rollouts: {args.random_fallback_eps} episodes")
        for _ in range(int(args.random_fallback_eps)):
            s = env.reset()
            done, ep_rew, steps = False, 0.0, 0
            max_steps = env.num_usvs * env.num_tasks + 5
            info = {}
            while not done and steps < max_steps:
                edges = compute_lookahead_edges(s, env.map_size, device=device)
                a, _, _ = agent.get_action(s, edges, epoch=0, max_epoch=1, deterministic=True)
                s, r, done, info = env.step(a)
                ep_rew += float(r)
                steps += 1
            makespans.append(info.get('makespan', 0.0))
            rewards.append(ep_rew)

    print(f"Avg makespan: {np.mean(makespans):.3f}  |  Avg reward(Δmakespan): {np.mean(rewards):.3f}")
    print(f"All makespan: {np.round(np.array(makespans), 3).tolist()}")
    print(f"All reward:   {np.round(np.array(rewards), 3).tolist()}")

if __name__ == "__main__":
    main()
