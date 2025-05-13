import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 设置图形大小和风格
plt.figure(figsize=(15, 10))
plt.style.use('seaborn-v0_8-whitegrid')

# 创建画布
ax = plt.gca()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')  # 隐藏坐标轴

# 颜色定义
colors = {
    'input': '#AED6F1',  # 输入/输出浅蓝色
    'conv': '#5DADE2',   # 卷积层蓝色
    'down': '#F5B041',   # 下采样橙色
    'up': '#58D68D',     # 上采样绿色
    'embed': '#BB8FCE',  # 嵌入层紫色
    'out': '#EC7063'     # 输出层红色
}

# 添加方块函数
def add_box(x, y, width, height, label, color, fontsize=10):
    rect = patches.Rectangle((x, y), width, height, linewidth=1, 
                            edgecolor='black', facecolor=color, alpha=0.7)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, 
            ha='center', va='center', fontsize=fontsize, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

# 添加箭头函数
def add_arrow(start, end, arrowstyle='->', color='black', linewidth=1.5, linestyle='-', connectionstyle="arc3,rad=0.1"):
    ax.annotate('', xy=end, xytext=start, 
                arrowprops=dict(arrowstyle=arrowstyle, color=color, 
                                lw=linewidth, linestyle=linestyle,
                                connectionstyle=connectionstyle))

# 添加文本标签
def add_label(x, y, text, fontsize=8):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

# 绘制所有组件

# 输入和输出模块
add_box(45, 90, 10, 6, "Input x\n(b, in_ch, h, w)", colors['input'])
add_box(45, 4, 10, 6, "Output\n(b, in_ch, h, w)", colors['out'])

# 主干下采样路径
add_box(45, 80, 10, 6, "init_conv\nResidualConvBlock", colors['conv'])
add_box(45, 70, 10, 6, "down1\nUnetDown", colors['down'])
add_box(45, 60, 10, 6, "down2\nUnetDown", colors['down'])
add_box(45, 50, 10, 6, "to_vec\nAvgPool + GELU", colors['conv'])

# 嵌入模块 (右侧)
add_box(70, 65, 12, 5, "timeembed1\nEmbedFC(1→2*n_feat)", colors['embed'])
add_box(70, 58, 12, 5, "timeembed2\nEmbedFC(1→n_feat)", colors['embed'])
add_box(70, 51, 12, 5, "contextembed1\nEmbedFC(n_cfeat→2*n_feat)", colors['embed'])
add_box(70, 44, 12, 5, "contextembed2\nEmbedFC(n_cfeat→n_feat)", colors['embed'])

# 输入到嵌入的箭头
add_arrow((55, 88), (70, 65), connectionstyle="arc3,rad=-0.2")
add_label(63, 75, "timestep t", fontsize=9)
add_arrow((55, 88), (70, 51), connectionstyle="arc3,rad=-0.3")
add_label(63, 68, "context c", fontsize=9)

# 主干上采样路径
add_box(45, 40, 10, 6, "up0\nConvTranspose2d", colors['up'])
add_box(45, 30, 10, 6, "up1\nUnetUp", colors['up'])
add_box(45, 20, 10, 6, "up2\nUnetUp", colors['up'])
add_box(45, 12, 10, 6, "out\nConv2d+GN+ReLU+Conv2d", colors['conv'])

# 主要连接箭头 (下行)
add_arrow((50, 90), (50, 86))
add_arrow((50, 80), (50, 76))
add_arrow((50, 70), (50, 66))
add_arrow((50, 60), (50, 56))
add_arrow((50, 50), (50, 46))
add_arrow((50, 40), (50, 36))
add_arrow((50, 30), (50, 26))
add_arrow((50, 20), (50, 18))
add_arrow((50, 12), (50, 10))

# 特征融合
add_box(45, 45, 10, 3, "特征融合 (中央瓶颈)", colors['embed'], fontsize=8)

# 嵌入层到特征融合的箭头
add_arrow((70, 65), (55, 45), connectionstyle="arc3,rad=0.2")
add_arrow((70, 58), (55, 45), connectionstyle="arc3,rad=0.1")
add_arrow((70, 51), (55, 45), connectionstyle="arc3,rad=0")
add_arrow((70, 44), (55, 45), connectionstyle="arc3,rad=-0.1")

# U-Net 跳跃连接 (skip connections)
skip_color = 'red'
add_arrow((55, 80), (55, 20), arrowstyle='-|>', color=skip_color, 
          linestyle='--', connectionstyle="arc3,rad=-0.4")
add_arrow((55, 70), (55, 30), arrowstyle='-|>', color=skip_color, 
          linestyle='--', connectionstyle="arc3,rad=-0.3")

# 标注跳跃连接
add_label(65, 38, "Skip Connections", fontsize=10)

# 各模块维度标注
add_label(35, 83, "→ (b, n_feat, h, w)", fontsize=8)
add_label(35, 73, "→ (b, n_feat, h/2, w/2)", fontsize=8)
add_label(35, 63, "→ (b, 2*n_feat, h/4, w/4)", fontsize=8)
add_label(35, 43, "→ (b, 2*n_feat, h/4, w/4)", fontsize=8)
add_label(35, 33, "→ (b, n_feat, h/2, w/2)", fontsize=8)
add_label(35, 23, "→ (b, n_feat, h, w)", fontsize=8)

# 标题
plt.title("ContextUnet Architecture", fontsize=18, pad=20)

# 图例
legend_elements = [
    patches.Patch(facecolor=colors['input'], edgecolor='black', alpha=0.7, label='Input/Output'),
    patches.Patch(facecolor=colors['conv'], edgecolor='black', alpha=0.7, label='Conv Blocks'),
    patches.Patch(facecolor=colors['down'], edgecolor='black', alpha=0.7, label='Down Blocks'),
    patches.Patch(facecolor=colors['up'], edgecolor='black', alpha=0.7, label='Up Blocks'),
    patches.Patch(facecolor=colors['embed'], edgecolor='black', alpha=0.7, label='Embedding Blocks')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('contextunet_architecture.png', dpi=300, bbox_inches='tight')
plt.show()