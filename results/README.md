# 实验数据

[返回首页](../README.md) · [实验口径](../docs/experiments.md)

5 个 UTF-8 编码的 CSV 文件分别提供检测精度、参数配置、框架验证、降维对比和消融实验数据。

| 文件 | 内容 | 数据行数 |
| --- | --- | --- |
| [table1_detection.csv](table1_detection.csv) | 3 类模型 × 5 种尺度的检测指标 | 15 |
| [table2_parameters.csv](table2_parameters.csv) | 粒球与三支划分参数 | 7 |
| [table3_framework.csv](table3_framework.csv) | 框架机制验证 | 3 |
| [table4_projection.csv](table4_projection.csv) | PCA、UMAP、t-SNE 对比 | 3 |
| [table5_ablation.csv](table5_ablation.csv) | GBC 模块消融 | 3 |

字段名中含 percent 的列使用百分数，例如 10.05 表示 10.05%。coverage_at_risk 表示高风险样本的确信区覆盖比例，fpr_95_reported 对应实验表中的 FPR-95% 指标。“-”表示该项无对应数据。指标公式见[方法说明](../docs/method.md)，评估步骤见[实验流程](../docs/reproducibility.md)。
