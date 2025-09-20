import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据准备
algorithms = [
    'usv_task_random_planner',
    'usv_lowest_battery_first',
    'usv_highest_battery_first',
    'task_farthest_distant_first',
    'task_nearest_distant_first',
    'USV-HGNN-PPO'
]

completion_times = [187.78, 207.04, 206.90, 437.53, 158.38, 119.37]

# 创建画布
plt.figure(figsize=(12, 6))

# 绘制柱状图
x = np.arange(len(algorithms))
width = 0.6
bars = plt.bar(x, completion_times, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])

# 添加标题和标签
plt.title('5X24 100 index', fontsize=16)
plt.xlabel('算法', fontsize=14)
plt.ylabel('总完成时间', fontsize=14)

# 设置x轴刻度和标签
plt.xticks(x, algorithms, rotation=45, ha='right', fontsize=12)

# 在柱状图上添加具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height:.2f}',
             ha='center', va='bottom', fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 可选：保存图形
# plt.savefig('completion_time_comparison.png', dpi=300, bbox_inches='tight')
