# 实验数据

[返回首页](../README.md) · [实验口径](../docs/experiments.md)

5 个 CSV 分别对应原稿表 1–5，使用 UTF-8 编码。数值按原表保留，合并单元格中的模型和阶段名称已补全。

| 文件 | 内容 | 数据行数 |
| --- | --- | --- |
| [table1_detection.csv](table1_detection.csv) | 3 类模型 × 5 种尺度的检测指标 | 15 |
| [table2_parameters.csv](table2_parameters.csv) | 粒球与三支划分参数 | 7 |
| [table3_framework.csv](table3_framework.csv) | 框架机制验证 | 3 |
| [table4_projection.csv](table4_projection.csv) | PCA、UMAP、t-SNE 对比 | 3 |
| [table5_ablation.csv](table5_ablation.csv) | GBC 模块消融 | 3 |

字段名中含 percent 的列沿用百分数，例如 10.05 表示 10.05%。coverage_at_risk 沿用原始数值；fpr_95_reported 的单位和阈值口径需要原始评测代码确认。“-”表示未报告，不能按零处理。类别 AP 的 IoU 口径、VSF 汇总方式等信息见[复现说明](../docs/reproducibility.md)。
