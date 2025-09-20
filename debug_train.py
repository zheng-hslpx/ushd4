"""
调试版本的 USV HGNN PPO 训练代码
修正HGNN模型边特征维度配置问题
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
import traceback
import time
import inspect

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 修正导入路径
from usv_agent.usv_env import USVEnv
from usv_agent.hgnn_model import HeterogeneousGNN
from usv_agent.ppo_policy import EnhancedPPOAgent, Memory
from usv_agent.data_generator import USVTaskDataGenerator
from utils.vis_manager import VisualizationManager

def debug_print(message, level=0):
    """调试打印函数"""
    indent = "  " * level
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{timestamp}] {indent}{message}")

def check_tensor_info(tensor, name):
    """检查张量信息"""
    if tensor is None:
        return f"{name}: None"
    elif isinstance(tensor, torch.Tensor):
        return f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, requires_grad={tensor.requires_grad}"
    else:
        return f"{name}: type={type(tensor)}, value={tensor}"

def check_state_info(state, name="state"):
    """检查状态信息"""
    if isinstance(state, dict):
        info = f"{name} (dict):"
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                info += f"\n  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}"
            elif isinstance(value, np.ndarray):
                info += f"\n  {key}: shape={value.shape}, dtype={value.dtype}"
            else:
                info += f"\n  {key}: type={type(value)}, value={value}"
        return info
    else:
        return f"{name}: type={type(state)}"

def inspect_hgnn_model():
    """检查HGNN模型的构造函数和edge_proj层"""
    debug_print("检查HGNN模型结构", 2)
    try:
        # 检查HeterogeneousGNN构造函数
        sig = inspect.signature(HeterogeneousGNN.__init__)
        debug_print(f"HeterogeneousGNN.__init__ 参数: {list(sig.parameters.keys())}", 2)
        
        # 创建一个临时模型来检查edge_proj层
        temp_config = {
            'usv_dim': 3,
            'task_dim': 4,
            'edge_dim': 1,  # 先尝试1
            'embed_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'hidden_dim': 128,
            'output_dim': 64
        }
        
        temp_hgnn = HeterogeneousGNN(temp_config)
        
        # 检查模型结构
        debug_print("HGNN模型结构:", 2)
        for name, module in temp_hgnn.named_modules():
            if 'edge_proj' in name:
                debug_print(f"  {name}: {module}", 2)
        
        return temp_config
        
    except Exception as e:
        debug_print(f"检查HGNN模型失败: {e}", 2)
        traceback.print_exc()
        return None

def compute_lookahead_edges(state, map_size, device, edge_dim=3):
    """
    修正版本的边计算函数
    根据HGNN模型的实际需求调整边特征维度
    """
    debug_print(f"计算 lookahead edges, edge_dim={edge_dim}", 3)
    
    # 检查状态中是否有位置信息
    if isinstance(state, dict):
        if 'usv_features' in state and 'task_features' in state:
            usv_feats = state['usv_features']  # [U, 3] -> [x, y, available_time]
            task_feats = state['task_features']  # [T, 4] -> [x, y, processing_time, is_active]
            
            # 转换为torch tensor
            if isinstance(usv_feats, np.ndarray):
                usv_feats = torch.from_numpy(usv_feats).to(device)
            if isinstance(task_feats, np.ndarray):
                task_feats = torch.from_numpy(task_feats).to(device)
            
            # 获取维度
            num_usvs = usv_feats.shape[0]
            num_tasks = task_feats.shape[0]
            
            # 创建边特征矩阵: [U, T, edge_dim]
            edges = torch.zeros((num_usvs, num_tasks, edge_dim), device=device)
            
            for i in range(num_usvs):
                for j in range(num_tasks):
                    # 计算USV和任务之间的距离
                    usv_pos = usv_feats[i, :2]  # x, y
                    task_pos = task_feats[j, :2]  # x, y
                    dist = torch.norm(usv_pos - task_pos)
                    
                    # 根据edge_dim填充边特征
                    if edge_dim >= 1:
                        edges[i, j, 0] = dist  # 距离
                    if edge_dim >= 2:
                        edges[i, j, 1] = usv_feats[i, 2]  # USV可用时间
                    if edge_dim >= 3:
                        edges[i, j, 2] = task_feats[j, 2]  # 任务处理时间
                    if edge_dim >= 4:
                        edges[i, j, 3] = task_feats[j, 3]  # 任务是否激活
                    # 如果edge_dim > 4，其余填0
                    for k in range(4, edge_dim):
                        edges[i, j, k] = 0.0
            
            return edges
        else:
            # 如果没有特征信息，创建随机边
            debug_print("状态中没有特征信息，使用随机边", 3)
            return torch.rand((4, 6, edge_dim), device=device)  # 假设4个USV，6个任务
    else:
        debug_print("状态不是字典格式，使用默认边", 3)
        return torch.rand((4, 6, edge_dim), device=device)

def debug_compute_lookahead_edges(state, map_size, device, edge_dim=3):
    """调试版本的边计算函数"""
    debug_print("开始计算 lookahead edges", 2)
    try:
        start_time = time.time()
        edges = compute_lookahead_edges(state, map_size, device, edge_dim)
        end_time = time.time()
        debug_print(f"边计算完成，耗时: {end_time - start_time:.4f}s", 2)
        debug_print(check_tensor_info(edges, "edges"), 2)
        return edges
    except Exception as e:
        debug_print(f"边计算失败: {e}", 2)
        traceback.print_exc()
        raise

def debug_get_action(agent, state, edges, epoch, max_epoch):
    """调试版本的动作获取函数"""
    debug_print("开始获取动作", 2)
    try:
        start_time = time.time()
        
        # 检查agent是否有get_action方法
        if hasattr(agent, 'get_action'):
            action, logprob, value = agent.get_action(
                state, edges, epoch=epoch, max_epoch=max_epoch, deterministic=False
            )
        elif hasattr(agent, 'select_action'):
            # 如果是select_action方法
            action, logprob, value = agent.select_action(state, edges)
        else:
            # 简单的随机动作作为fallback
            debug_print("智能体没有get_action方法，使用随机动作", 2)
            if isinstance(state, dict) and 'action_mask' in state:
                # 使用action_mask选择有效动作
                action_mask = state['action_mask']
                valid_actions = np.where(action_mask == 1)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = 0  # 如果没有有效动作，选择0
            else:
                action = np.random.randint(0, 24)  # 4*6=24个可能的动作
            
            logprob = torch.log(torch.tensor(1.0/24.0))
            value = torch.tensor(0.0)
        
        end_time = time.time()
        debug_print(f"动作获取完成，耗时: {end_time - start_time:.4f}s", 2)
        debug_print(f"action: {action} (type: {type(action)})", 2)
        debug_print(f"logprob: {logprob} (type: {type(logprob)})", 2)
        debug_print(f"value: {value} (type: {type(value)})", 2)
        return action, logprob, value
    except Exception as e:
        debug_print(f"动作获取失败: {e}", 2)
        traceback.print_exc()
        raise

def debug_env_step(env, action):
    """调试版本的环境步进函数"""
    debug_print(f"开始环境步进，动作: {action}", 2)
    try:
        start_time = time.time()
        next_state, reward, done, info = env.step(action)
        end_time = time.time()
        debug_print(f"环境步进完成，耗时: {end_time - start_time:.4f}s", 2)
        debug_print(f"reward: {reward} (type: {type(reward)})", 2)
        debug_print(f"done: {done} (type: {type(done)})", 2)
        debug_print(f"info: {info}", 2)
        debug_print(check_state_info(next_state, "next_state"), 2)
        return next_state, reward, done, info
    except Exception as e:
        debug_print(f"环境步进失败: {e}", 2)
        traceback.print_exc()
        raise

def debug_memory_add(mem, state, action, logprob, reward, done, value, edges):
    """调试版本的内存添加函数"""
    debug_print("开始添加到内存", 2)
    try:
        debug_print("检查输入数据:", 3)
        debug_print(check_state_info(state), 3)
        debug_print(f"action: {action} (type: {type(action)})", 3)
        debug_print(f"logprob: {logprob} (type: {type(logprob)})", 3)
        debug_print(f"reward: {reward} (type: {type(reward)})", 3)
        debug_print(f"done: {done} (type: {type(done)})", 3)
        debug_print(f"value: {value} (type: {type(value)})", 3)
        debug_print(check_tensor_info(edges, "edges"), 3)
        
        # 检查Memory对象的属性
        debug_print(f"Memory对象属性: {[attr for attr in dir(mem) if not attr.startswith('_')]}", 3)
        
        # 使用Memory的add方法
        if hasattr(mem, 'add'):
            mem.add(state, action, logprob, reward, done, value, edges)
        else:
            # 手动添加到内存
            mem.states.append(state)
            
            # 检查edges属性名
            if hasattr(mem, 'usv_task_edges'):
                mem.usv_task_edges.append(edges.detach().cpu())
            elif hasattr(mem, 'edges'):
                mem.edges.append(edges.detach().cpu())
            else:
                debug_print("Memory对象没有edges相关属性，跳过", 3)
            
            mem.actions.append(action)
            mem.logprobs.append(logprob)
            mem.values.append(value)
            mem.rewards.append(float(reward))
            mem.is_terminals.append(done)
        
        debug_print(f"内存添加成功，当前大小: {len(mem.states)}", 2)
        
    except Exception as e:
        debug_print(f"内存添加失败: {e}", 2)
        traceback.print_exc()
        raise

def debug_ppo_update(agent, memory):
    """调试版本的PPO更新函数"""
    debug_print("开始PPO更新", 1)
    try:
        debug_print(f"内存大小: {len(memory.states)}", 2)
        start_time = time.time()
        
        # 检查agent是否有update方法
        if hasattr(agent, 'update'):
            loss_info = agent.update(memory)
        else:
            debug_print("智能体没有update方法，跳过更新", 2)
            loss_info = {"message": "no update method"}
        
        end_time = time.time()
        debug_print(f"PPO更新完成，耗时: {end_time - start_time:.4f}s", 2)
        debug_print(f"损失信息: {loss_info}", 2)
        return loss_info
    except Exception as e:
        debug_print(f"PPO更新失败: {e}", 2)
        traceback.print_exc()
        raise

def main():
    debug_print("=== 开始调试程序 ===")
    
    try:
        # 1. 设置设备
        debug_print("1. 设置计算设备")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        debug_print(f"使用设备: {device}")
        if device.type == 'cuda':
            debug_print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            debug_print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        
        # 2. 创建环境
        debug_print("2. 创建环境")
        try:
            # 根据USVEnv的构造函数，需要传递env_config字典
            env_config = {
                'num_usvs': 4,
                'num_tasks': 6,
                'map_size': [10.0, 10.0],  # 改为列表格式
                'usv_speed': 5.0
            }
            env = USVEnv(env_config)
            debug_print(f"环境创建成功: {env.num_usvs} USVs, {env.num_tasks} tasks, map_size={env.map_size}")
        except Exception as e:
            debug_print(f"环境创建失败: {e}")
            traceback.print_exc()
            return
        
        # 3. 检查HGNN模型结构
        debug_print("3. 检查HGNN模型结构")
        hgnn_config = inspect_hgnn_model()
        if hgnn_config is None:
            return
        
        # 4. 尝试不同的edge_dim值
        debug_print("4. 尝试不同的edge_dim配置")
        edge_dims_to_try = [3, 1, 4, 2]  # 按优先级尝试不同的边特征维度
        
        hgnn = None
        edge_dim = None
        
        for test_edge_dim in edge_dims_to_try:
            debug_print(f"尝试 edge_dim = {test_edge_dim}", 2)
            try:
                # 创建HGNN配置
                hgnn_config = {
                    'usv_dim': 3,      # USV特征维度 [x, y, available_time]
                    'task_dim': 4,     # Task特征维度 [x, y, processing_time, is_active]
                    'edge_dim': test_edge_dim,     # 边特征维度
                    'embed_dim': 64,   # 嵌入维度
                    'num_layers': 2,   # HGNN层数
                    'num_heads': 4,    # 注意力头数
                    'dropout': 0.1,    # Dropout率
                    'hidden_dim': 128, # 隐藏层维度
                    'output_dim': 64   # 输出维度
                }
                
                hgnn = HeterogeneousGNN(hgnn_config)
                hgnn = hgnn.to(device)
                edge_dim = test_edge_dim
                debug_print(f"HGNN模型创建成功，edge_dim={edge_dim}", 2)
                debug_print(f"HGNN参数数量: {sum(p.numel() for p in hgnn.parameters())}", 2)
                break
                
            except Exception as e:
                debug_print(f"edge_dim={test_edge_dim} 失败: {e}", 2)
                continue
        
        if hgnn is None:
            debug_print("所有edge_dim配置都失败", 1)
            return
        
        # 5. 创建智能体
        debug_print("5. 创建智能体")
        try:
            # 创建模型配置
            model_cfg = {
                'embed_dim': 64,
                'n_hidden': 2,
                'n_latent': 128,
                'dropout': 0.1,
                'lr': 3e-4,
                'eps_clip': 0.2,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'max_grad_norm': 0.5,
                'ppo_epochs': 4,
                'batch_size': 64,
                'exploration_eps': 0.1
            }
            
            # 使用正确的构造函数
            agent = EnhancedPPOAgent(hgnn, model_cfg)
            debug_print("EnhancedPPOAgent创建成功")
            debug_print(f"Agent参数数量: {sum(p.numel() for p in agent.parameters()) if hasattr(agent, 'parameters') else 'N/A'}")
            
        except Exception as e:
            debug_print(f"EnhancedPPOAgent创建失败: {e}")
            traceback.print_exc()
            return
        
        # 6. 初始化记录
        episode_rewards = []
        
        # 7. 开始训练循环
        debug_print("7. 开始训练循环")
        
        for ep in range(1, 4):  # 只运行3个episode用于调试
            debug_print(f"--- Episode {ep} 开始 ---")
            
            try:
                # 7.1 重置环境
                debug_print("重置环境", 1)
                state = env.reset()
                debug_print(check_state_info(state, "初始状态"), 1)
                
                # 7.2 初始化episode变量
                memory = Memory()
                done = False
                steps = 0
                episode_reward = 0.0
                max_steps = 20  # 限制步数用于调试
                debug_print(f"Episode初始化完成，最大步数: {max_steps}", 1)
                
                # 7.3 步骤循环
                debug_print("开始步骤循环", 1)
                step_start_time = time.time()
                
                while not done and steps < max_steps:
                    debug_print(f"Step {steps}", 1)
                    
                    # 检查GPU内存
                    if device.type == 'cuda':
                        debug_print(f"GPU内存使用: {torch.cuda.memory_allocated() / 1e6:.1f} MB", 2)
                    
                    # 计算边 - 使用确定的edge_dim
                    edges = debug_compute_lookahead_edges(state, env.map_size, device, edge_dim)
                    
                    # 获取动作
                    action, logprob, value = debug_get_action(agent, state, edges, ep, 10)
                    
                    # 环境步进
                    next_state, reward, done, info = debug_env_step(env, action)
                    
                    # 添加到内存
                    debug_memory_add(memory, state, action, logprob, reward, done, value, edges)
                    
                    # 更新状态
                    state = next_state
                    episode_reward += float(reward)
                    steps += 1
                    
                    debug_print(f"Step {steps} 完成，累计奖励: {episode_reward:.4f}", 1)
                    
                    # 安全检查：如果单步耗时过长，退出
                    if time.time() - step_start_time > 10:  # 10秒超时
                        debug_print("单步耗时过长，强制退出", 1)
                        break
                    
                    step_start_time = time.time()
                
                debug_print(f"步骤循环结束: steps={steps}, done={done}, reward={episode_reward:.4f}", 1)
                
                # 7.4 PPO更新
                if len(memory.states) > 0:
                    loss_info = debug_ppo_update(agent, memory)
                else:
                    debug_print("跳过PPO更新（内存为空）", 1)
                
                # 7.5 记录
                episode_rewards.append(episode_reward)
                debug_print(f"Episode {ep} 完成，奖励: {episode_reward:.4f}", 1)
                
            except KeyboardInterrupt:
                debug_print("用户中断")
                break
            except Exception as e:
                debug_print(f"Episode {ep} 异常: {e}")
                traceback.print_exc()
                break
        
        debug_print("=== 调试程序完成 ===")
        debug_print(f"完成的episodes: {len(episode_rewards)}")
        if episode_rewards:
            debug_print(f"平均奖励: {np.mean(episode_rewards):.4f}")
        
    except Exception as e:
        debug_print(f"程序异常: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
