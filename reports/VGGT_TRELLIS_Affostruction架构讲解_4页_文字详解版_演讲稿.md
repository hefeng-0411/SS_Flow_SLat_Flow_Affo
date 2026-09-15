# VGGT + TRELLIS + Affostruction 架构讲解——文字详解版演讲稿

## 第 1 页：总体方法

当前架构的核心变化，是把图像条件从全局语义提升为三维空间条件。VGGT 提供相机、深度、点图和置信度，TRELLIS 提供完整三维对象的生成先验。项目在两者之间引入 Affostruction 风格的空间 token 和条件流：SS 把像素证据附着到三维 voxel，SLat 再从 active voxel 回到所有图像中读取对应语义与纹理。这样形成 Pixel 到 Voxel，再回到 Pixel-aligned Evidence 的双向闭环。项目只吸收适合图像三维重建的空间条件化、cross-attention 和 flow matching，不采用文本或 affordance prompt。

## 第 2 页：SS 微调

SS 阶段先把多视图 RGBD 变成与三维位置对应的条件。冻结 DINO 提取空间 patch 特征，深度与相机把像素反投影到 canonical 空间并量化到十六立方 voxel；同一 voxel 内通过平均二维位置进行特征采样，再跨有效视图平均。LayerNorm 和三维位置编码使 token 同时具有语义与位置。官方 Affostruction SparseStructureFlowModel 有十二层、宽度七百六十八，全部参数微调，条件 token 通过 cross-attention 进入每个生成 block。训练使用 Conditional Flow Matching，仅拟合正确的 latent velocity；推理利用 EMA 权重、CFG 和 ODE 积分得到 active support。

## 第 3 页：SLat 微调

SLat 阶段把 SS active voxel 投影回所有输入视图，在同一像素位置采样高层语义和低层纹理，并结合前景、遮挡、深度、法向角度和 VGGT confidence 计算逐 voxel、逐视图权重。融合后的空间 token 输入独立 Affostruction SLatFlowModel。这里没有覆盖 TRELLIS 原生 SLat flow，而是冻结它作为 base velocity。新模型只学习 residual，残差按基础速度 RMS 缩放，通过 tanh 限幅并乘 voxel confidence。于是图像证据只在可靠区域发挥作用，而遮挡或未观测区域继续保留 TRELLIS 的生成先验。

## 第 4 页：训练与推理闭环

训练阶段严格划分梯度边界。SS 只更新官方 Affostruction Flow，DINO 和 RGBD 条件冻结；SLat 只更新 residual flow 和 conditioner，TRELLIS 原生 SLat flow 冻结。SLat 除了速度残差，还通过 endpoint、prior preservation 和解码资产损失约束最终三维结果。推理阶段按 frame ID 同步全部输入，SS 先生成 active support，SLat 再在每个 ODE 步把 frozen base 与 gated residual 相加，最后交给 TRELLIS decoder。训练与推理必须共享 canonical frame、相机约定和 mask 契约。整套设计的最终原则，是让 VGGT 精确约束可观测区域，让 TRELLIS 完成不确定区域。
