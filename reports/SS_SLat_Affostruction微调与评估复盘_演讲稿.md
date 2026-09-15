# SS / SLat Affostruction 微调与评估复盘——演讲稿

## 第 1 页：标题与结论

本页先给出整场汇报的结论。项目由两个阶段组成：SS 负责生成稀疏结构支撑，SLat 负责在支撑坐标上生成外观和局部几何属性。当前代码已经从旧版的大骨干直接改写，转向官方 SS 全量流模型，以及冻结 TRELLIS 先验的 SLat 残差模型。需要特别强调，后面的评估数据来自旧版检查点，所以这些结果是重构动机，而不是当前新架构的最终成绩。

## 第 2 页：端到端流程

端到端流程可以压缩为五步。第一步读取严格按 frame ID 对齐的稀疏多视图数据。第二步由 VGGT 提供相机、深度、点图和置信度，DINO 提供语义特征。第三步 SS 在 16 立方网格上生成 active support。第四步 SLat 把这些 active voxel 投影回所有输入视图，采样语义与纹理特征并融合，然后仅预测相对原始 TRELLIS 的速度残差。第五步由 TRELLIS 解码器生成 Gaussian 和 mesh。这个分工避免让一个模块同时承担拓扑、材质和多视图一致性。

## 第 3 页：SS 阶段改进

SS 的旧版问题并不是没有学习，而是学习目标发生了偏移。旧模型有二十四层、宽度一千零二十四，并同时承受 CFM、occupancy、silhouette 和 depth 多种梯度。日志显示总损失下降主要来自 occupancy，而核心 CFM 后期反而更差。当前设计改为 Affostruction 官方的十二层、宽度七百六十八模型。DINO 只提供冻结条件，SS Flow 全参数训练，损失严格使用条件流匹配 MSE。这样训练目标重新变成生成正确的 latent velocity，而不是单纯提高低分辨率 occupancy。

## 第 4 页：SLat 阶段改进

SLat 的旧架构直接更新原生 TRELLIS 大骨干，这是外观退化的主要风险。同时旧训练使用 GT support，而组合推理使用预测 SS support，训练和推理分布不一致。当前方案冻结原生 TRELLIS，把它作为基础速度。新的 Affostruction 稀疏流只预测残差。每个 active voxel 会投影到所有输入视图，分别采样高层 DINO 语义和低层 RGB 纹理，再使用置信度、前景、遮挡、深度和角度一致性融合。最终残差经过 tanh 限幅并乘以置信度，因此低可信或未观测区域会自动回退到原始 TRELLIS。

## 第 5 页：评估结果

评估覆盖二百五十一个对象，四组全部成功。SS-only 的 Asset Chamfer Distance 下降约百分之十五，SS IoU 从零点一二提高到零点三二四，说明 SS 确实学到了更接近 GT 的支撑结构。但是图像质量明显下降：SS-only 的 PSNR 下降六点六 dB，组合模型下降十点二二 dB。前景指标也同步恶化，因此不是背景造成的假象。SLat-only 的几何改善不到百分之一，却损失七点三八 dB PSNR。组合模型比 SS-only 还差，表明旧 SLat 不能修复 SS，反而继续破坏 latent appearance。

## 第 6 页：优化路线

优化必须按顺序推进。第一，不继续使用旧检查点，而是从当前新架构重新训练。第二，SLat 后期必须使用当前 SS 的真实预测 support，消除 teacher forcing 和推理之间的分布差异。第三，拆解并校准置信度各因子，先验证相机和深度几何，再调整阈值。第四，提高 decoded asset 监督频率和视图数，并把前景感知指标直接纳入验证。第五，不能再只评估 last checkpoint，要比较 EMA、best 和 last，并在完整验证集上选模型。最终验收标准不是训练 loss 下降，而是 SS+SLat 在保持几何收益的同时，显著优于 SS-only 的前景 PSNR、LPIPS 和轮廓指标。
