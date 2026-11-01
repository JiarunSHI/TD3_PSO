import numpy as np
import matplotlib.pyplot as plt

# ========== 全局字体 Times New Roman, 8 pt ==========
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

# ========== 数据 ==========
paths = ['Path1', 'Path2', 'Path3', 'Path4']
fitness_names = ['Fitness', 'Fitness2', 'Fitness3', 'Fitness4']
data = np.array([
    [9.8130, 8.5880, 7.9110, 8.5110],
    [9.9480, 8.3250, 8.9570, 9.8650],
    [10.5350, 9.2960, 7.7810, 8.8390],
    [9.8420, 9.3430, 13.5750, 9.0580]
])

# ========== 创建图形：上方柱状图，下方表格 ==========
fig, (ax_bar, ax_table) = plt.subplots(2, 1, figsize=(8, 6),
                                       gridspec_kw={'height_ratios': [3, 1]})

# ---------- 柱状图 ----------
x = np.arange(len(paths))
width = 0.2
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for i, (name, color) in enumerate(zip(fitness_names, colors)):
    ax_bar.bar(x + i * width, data[i], width, label=name,
               color=color, edgecolor='black', lw=0.5)

ax_bar.set_xlabel('Path Index')
ax_bar.set_ylabel('Fitness Value')
ax_bar.set_xticks(x + width * 1.5)
ax_bar.set_xticklabels(paths)
ax_bar.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
ax_bar.grid(True, axis='y', alpha=0.3)

# ---------- 表格 ----------
ax_table.axis('off')

col_labels = [''] + paths
row_labels = fitness_names
cell_text = [[f'{val:.4f}' for val in row] for row in data]

table = ax_table.table(cellText=cell_text,
                       rowLabels=row_labels,
                       colLabels=col_labels,
                       loc='center',
                       cellLoc='center')

# 表格样式
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

for key, cell in table.get_celld().items():
    cell.set_edgecolor('black')
    cell.set_linewidth(0.5)
    if key[0] == 0:          # 标题行
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e6e6e6')
    else:
        cell.set_facecolor('white')

# 移除所有图片内标题（IEEE 要求）
plt.tight_layout()
plt.show()