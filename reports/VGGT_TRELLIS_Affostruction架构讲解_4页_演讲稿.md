# VGGT + TRELLIS + Affostruction 架构讲解——4页演讲稿

## 第 1 页：总体思路

当前架构吸收 Affostruction 的核心并不是增加一个普通 Adapter，而是改变条件进入生成模型的方式。VGGT 输出相机、深度、点图和置信度，SS 阶段把二维像素证据转换成十六立方三维条件，生成 active support；SLat 阶段从六十四立方 active voxel 再投影回图像，读取对应的语义与纹理。Affostruction 的空间 token、三维位置编码、cross-attention 和 Conditional Flow Matching 被保留，但文本或 affordance 条件没有被采用。系统最终通过置信度决定由观测证据还是 TRELLIS 先验控制某个区域。

## 第 2 页：SS 条件流

SS 的条件构造分为特征提取、三维反投影和跨视图聚合。冻结 DINO 从每个输入视图生成一千零二十四维 patch feature；VGGT 或数据集深度结合相机内外参把像素恢复到 canonical 三维空间，再量化到十六立方 voxel。同一 voxel 中的像素先确定平均二维采样位置，通过 grid_sample 读取 DINO 特征，之后在所有有效视图间平均。聚合结果经过 LayerNorm 并加入三维位置编码，只有被观测到的 voxel 被压缩为条件 token。官方 Affostruction SS Flow 使用十二层、宽度七百六十八的 Transformer，通过 cross-attention 读取这些条件，训练目标为 Conditional Flow Matching 速度 MSE，推理通过 ODE 和 CFG 得到 active support。

## 第 3 页：SLat 可信残差流

SLat 阶段从 SS 的 active voxel 出发，将每个三维坐标投影到所有有效输入视图，在相同像素位置采样高层 DINO 或 VGGT 特征以及浅层 RGB CNN 特征。高层特征负责部件语义，低层特征保留颜色和局部纹理。每个视图的融合权重由 VGGT confidence、可见性、前景遮挡、深度一致性和法向入射角共同决定。融合后的 token 通过 cross-attention 输入 Affostruction SLatFlowModel。这个稀疏流不替换 TRELLIS，而是预测 residual。TRELLIS 原生 SLat velocity 完全冻结，残差先按 base velocity RMS 确定尺度，再用 tanh 限幅并乘 voxel confidence，因此有证据区域得到校正，弱证据区域保留原始先验。

## 第 4 页：训练与推理闭环

训练分成 SS 和 SLat 两个阶段。SS 中只更新官方 Affostruction SS Flow，DINO 和 RGBD conditioner 冻结，使用 CFM 速度 MSE 和 EMA。得到 SS 模型后生成 active support。SLat 中冻结 TRELLIS 原生 SLat flow，把它作为 base teacher，训练独立 residual flow 和 conditioner，使最终速度对齐目标速度，同时使用 endpoint、低置信先验保持和解码资产监督。推理时 SS 先生成 support，SLat 在每个 ODE 步将 frozen base 与 gated residual 相加，再交给 TRELLIS decoder。整个闭环必须保持 frame ID、相机约定、canonical 坐标以及条件 mask 一致。最终方法可以概括为 Pixel 到 Voxel，再从 Voxel 回到像素证据，最后形成受控 Structured Latent。
