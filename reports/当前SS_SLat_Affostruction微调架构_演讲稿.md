# 当前 SS / SLat Affostruction 微调架构——演讲稿

## 第 1 页：标题与结论

本页给出整体架构的核心定位。项目由两个阶段组成：SS 负责生成稀疏结构支撑，SLat 负责在支撑坐标上生成外观和局部几何属性。在 VGGT 加 TRELLIS 基础上，系统加入 Affostruction 官方 SS 全量条件流，以及冻结 TRELLIS 原生 SLat 先验的像素对齐残差流。整个设计的目标是把可观测几何注入模型，同时尽量不破坏 TRELLIS 的生成分布。

## 第 2 页：端到端流程

端到端流程可以压缩为五步。第一步读取严格按 frame ID 对齐的稀疏多视图数据。第二步由 VGGT 提供相机、深度、点图和置信度，DINO 提供语义特征。第三步 SS 在 16 立方网格上生成 active support。第四步 SLat 把这些 active voxel 投影回所有输入视图，采样语义与纹理特征并融合，然后仅预测相对原始 TRELLIS 的速度残差。第五步由 TRELLIS 解码器生成 Gaussian 和 mesh。这个分工避免让一个模块同时承担拓扑、材质和多视图一致性。

## 第 3 页：SS 微调算法

SS 阶段是在 VGGT 几何输出与 TRELLIS 稀疏结构生成之间加入显式三维条件。输入图像先经过冻结 DINO，得到一千零二十四维 patch token；对应深度和相机把像素反投影到 canonical 三维空间，并量化到十六立方 voxel。同一 voxel 内先聚合像素，再跨有效视图平均，之后进行 LayerNorm 并加入三维位置编码。紧凑的已观测 voxel token 通过 cross-attention 注入 Affostruction 官方 SparseStructureFlowModel。该模型有十二个 block、hidden 七百六十八，全部参数参与微调，训练目标是条件流速度，推理通过 ODE 采样得到 active support。

## 第 4 页：SLat 微调算法

SLat 阶段以 SS 给出的六十四立方 active voxel 为稀疏支撑。系统将每个 voxel 投影回所有有效相机，在对应像素同时采样高层 DINO 语义和浅层卷积提取的 RGB 纹理。每个视图的权重由 VGGT confidence、可见性、前景、深度一致性和入射角一致性共同决定，融合结果投影为一千零二十四维条件 token。一个独立的 Affostruction SLatFlowModel 根据这些条件预测速度残差。TRELLIS 原生 SLat flow 完全冻结，最终修正通过 tanh 限幅并乘以 voxel confidence，所以观测充分的位置得到几何和纹理校正，证据不足的位置保留 TRELLIS 先验。

## 第 5 页：训练目标与梯度边界

这一页说明梯度到底更新谁。SS 的训练路径与 Affostruction 官方条件流一致：从 clean latent 和高斯噪声构造 xt，目标速度为一减 sigma-min 乘噪声再减 x0，损失是预测速度和目标速度的均方误差。DINO 和条件构造冻结，SS Flow 全参数更新。SLat 以冻结 TRELLIS 的速度为基线，只训练独立稀疏残差流和多视图 conditioner。最终速度等于基础速度加置信度调制后的受限残差。损失同时约束 flow velocity、重建端点、低置信区域的先验保持，以及解码后的资产质量。数值实现使用混合精度前向、FP32 主权重、DDP 和梯度检查点。

## 第 6 页：推理闭环与总体思路

完整推理首先读取严格按 frame ID 对齐的多视图图像、mask、深度和相机。SS 条件流通过 CFG 和 EMA 权重采样 active support。随后将 active voxel 投影到所有有效视图，构造逐 voxel 的 SLat condition。SLat ODE 在每个采样步保留冻结的 TRELLIS base velocity，并叠加置信度门控的残差，最后解码成 Gaussian 和 mesh。总体思路是几何条件化生成：VGGT 为可观测区域提供确定性约束，TRELLIS 为遮挡和未观测区域提供生成先验，高层语义与低层纹理共同决定结构化潜变量，置信度负责在两者之间动态分配控制权。
