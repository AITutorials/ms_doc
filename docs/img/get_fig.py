
import matplotlib.pyplot as plt
# 设置背景风格
plt.switch_backend('Agg')

def plot(metric, metric_name, labels, fig_name):
    """
    绘制函数
    :param metric: 需要对比的评估指标列表
    :param metric: 需要对比的评估指标名称
    :param labels: 对比实验的名称列表，列表长度与metric相同
    :param: fig_nam: 对比图保存名称
    """
    # 初始化画布和轴
    fig, ax = plt.subplots()
    # 使用数据绘制柱状图，并且纵坐标范围为[0, 1]
    ax.bar([0,1], metric)
    # 设置纵坐标名称
    ax.set_ylabel(metric_name)
    # 设置横轴刻度
    ax.set_xticks([0,1])
    # 设置横轴名称
    ax.set_xticklabels(labels)
    # 加入表格线
    ax.yaxis.grid(True)
    # 设置布局
    plt.tight_layout()
    # 保存图像并关闭画布
    plt.savefig(fig_name)
    plt.close(fig)


# 两次的精度结果
precision = [0.882, 0.922]
# 两次的召回率结果
recall = [0.871, 0.898]

test1 = '使用自动超参数调优前'
test2 = '使用自动超参数调优后'

fig_name1 = "at_pre.png"
fig_name2 = "at_recall.png"


"""
precision = [0.922, 0.938]
recall = [0.898, 0.892]

test1 = "字粒度"
test2 = "词粒度"

fig_name1 = "gran_pre.png"
fig_name2 = "gran_recall.png"
"""

precision = [0.938, 0.932]
recall = [0.892, 0.895]

test1 = "数据增强前"
test2 = "数据增强后"

fig_name1 = "augm_pre.png"
fig_name2 = "augm_recall.png"


# ----

precision = [0.938, 0.951]
recall = [0.892, 0.927]

test1 = "迁移词向量前"
test2 = "迁移词向量后"

fig_name1 = "vec_pre.png"
fig_name2 = "vec_recall.png"



plot(precision, "精度", [test1, test2], fig_name1)
plot(recall, "召回率", [test1, test2], fig_name2)







