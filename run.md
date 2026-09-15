已完成代码迁移与本地验收；正式 4-GPU 训练尚未启动。当前 GPU 0/2 正在满载，请先确认相关任务结束，避免抢占显存。

验证结果：

- 完整测试：`205 passed, 1 skipped`
- 新 Affostruction 架构测试：全部通过
- SLat CPU dry-run：通过
- Kaolin 投影、GeomLoss Sinkhorn、反向梯度：通过
- 数据审计：SS/SLat 可训练对象各 `1226` 个
- 缺号图片按真实 frame ID 对齐并跳过缺失帧，不丢弃整个对象子集
- 扩展导入：`spconv / kaolin / nvdiffrast / geomloss / gsplat` 全部通过
- 官方实现边界依据：[Affostruction 论文](https://arxiv.org/html/2601.09211v2)、[官方源码](https://github.com/chrockey/Affostruction)。官方公开源码没有重建训练代码，因此 SS 按公开前向机制实现，SLat 是明确的对称数学外推。

入口为 [train_ddp_affostruction.sh](/mnt/sda/hf/MVG/Base/SS_Flow_SLat_Flow_Affo/train_ddp_affostruction.sh)。

## 1. 环境初始化

```bash
source /mnt/sda/hf/miniconda3/etc/profile.d/conda.sh
conda activate trellis

cd /mnt/sda/hf/MVG/Base/SS_Flow_SLat_Flow_Affo

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
```

确认四张 GPU 已空闲：

```bash
nvidia-smi
```

## 2. 依赖与扩展预检

```bash
python - <<'PY'
import importlib

modules = (
    "torch",
    "spconv.pytorch",
    "kaolin",
    "nvdiffrast.torch",
    "geomloss",
    "gsplat",
)

for module in modules:
    importlib.import_module(module)
    print("OK", module)
PY
```

## 3. 数据集完整审计

已经生成：

```text
outputs/affostruction_dataset_audit/stage2_train_uids.json
outputs/affostruction_dataset_audit/stage3_train_uids.json
outputs/affostruction_dataset_audit/validation_evaluation_uids.json
outputs/affostruction_dataset_audit/test_evaluation_uids.json
```

若要重新进行包含载荷读取的完整审计：

```bash
python scripts/inspect_meshfleet_dataset.py \
  --data_root /mnt/sda/hf/MVG/Base/dataset/c9028d206944a33af776f1b6967a6d82af385e97 \
  --output_dir outputs/affostruction_dataset_audit \
  --splits train,test \
  --min_train_views 1 \
  --min_eval_views 12 \
  --validate_payloads \
  --strict \
  --strict_scope manifests
```

`--min_train_views 1` 保证有图片缺号但至少存在一个有效视图的对象仍可参与训练。

## 4. 代码测试

完整测试：

```bash
python -m pytest -q
```

重点架构测试：

```bash
python -m pytest -q \
  tests/test_affostruction_migration.py \
  tests/test_stochastic_voxel_ss_flow.py
```

验证缺号图片索引对齐：

```bash
python -m pytest -q \
  tests/test_stochastic_voxel_ss_flow.py::test_dataset_gapped_render_ids_keep_image_camera_and_metadata_aligned
```

SLat CPU dry-run：

```bash
python scripts/train_geovis_slat.py \
  --config configs/geovis_slat.yaml \
  --dry_run true \
  --device cpu \
  --batch_size 1 \
  --num_views 2 \
  --image_size 32 \
  --output_dir outputs/smoke_slat_cpu
```

## 5. 多卡 smoke test

先测试 SS，执行 4 个更新并进行验证：

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
./train_ddp_affostruction.sh ss \
  --output-dir outputs/smoke_affostruction_ss \
  --max-steps 4 \
  --save-every 1 \
  --log-every 1 \
  --validate-every 2 \
  --validation-batches 1 \
  2>&1 | tee outputs/smoke_affostruction_ss.log
```

然后测试 SLat。第 4 步会真实执行 TRELLIS Gaussian 解码、gsplat 多视图渲染、Kaolin 重投影和 GeomLoss Sinkhorn：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
./train_ddp_affostruction.sh slat \
  --output_dir outputs/smoke_affostruction_slat \
  --steps 4 \
  --steps_are_total true \
  --minimum_dataset_passes 0 \
  --execution_mode smoke \
  --early_stop false \
  --save_every 1 \
  --visualize_every 0 \
  2>&1 | tee outputs/smoke_affostruction_slat.log
```

检查 smoke 输出：

```bash
tail -n 4 outputs/smoke_affostruction_ss/metrics.jsonl
tail -n 4 outputs/smoke_affostruction_slat/train_geovis_slat.jsonl

test -s outputs/smoke_affostruction_ss/stochastic_voxel_ss_flow_last.pt
test -s outputs/smoke_affostruction_slat/geovis_slat_adapter_last.pt
```

## 6. 正式 SS 训练

```bash
nohup bash -c '
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
./train_ddp_affostruction.sh ss \
  --output-dir outputs/affostruction_ss \
  --max-steps 10000 \
  --gradient-accumulation-steps 1 \
  --save-every 100 \
  --log-every 10 \
  --validate-every 500 \
  --validation-batches 1
' > outputs/affostruction_ss.log 2>&1 &

echo "SS training started in background. PID: $!"
```

SS 训练输出：

```text
outputs/affostruction_ss/stochastic_voxel_ss_flow_last.pt
outputs/affostruction_ss/metrics.jsonl
outputs/affostruction_ss/validation.jsonl
outputs/affostruction_ss/pathologies_rank*.jsonl
```

监控：

```bash
tail -F outputs/affostruction_ss/metrics.jsonl
```

正常日志应包含：

```text
architecture_version = affostruction_direct_ss_cross_attention_v1
optimizer_step_applied = true
alignment_mode
conditioning_voxel_count
geometry_condition_fraction
backbone_velocity_norm
```

## 7. SS 断点续训

```bash
nohup bash -c '
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
./train_ddp_affostruction.sh ss \
  --output-dir outputs/affostruction_ss \
  --resume outputs/affostruction_ss/stochastic_voxel_ss_flow_last.pt \
  --max-steps 10000
' >> outputs/affostruction_ss.log 2>&1 &

echo "SS resume started in background. PID: $!"
```

不要使用旧项目产生的 residual-adapter checkpoint。加载器会主动拒绝架构版本不一致的检查点。

## 8. 正式 SLat 训练

SLat 使用 GT sparse support 做训练期 teacher forcing；推理时使用 SS 预测 active voxels。这是离散 active-support 边界，不能通过连续反向传播跨越。

```bash
nohup bash -c '
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
./train_ddp_affostruction.sh slat \
  --output_dir outputs/affostruction_slat \
  --steps 1000 \
  --steps_are_total true \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --save_every 100 \
  --visualize_every 100 \
  --early_stop true \
  --minimum_dataset_passes 1
' > outputs/affostruction_slat.log 2>&1 &

echo "SLat training started in background. PID: $!"
```

该命令默认启用：

- 完整 SLat backbone 梯度
- BF16 autocast
- gradient checkpointing
- 多视图 DINO 高层特征
- 浅层 RGB 特征
- Kaolin active-voxel 投影
- 置信度、遮挡、角度及深度一致性融合

## 9. 四卡推理评估与断点恢复

完整四种消融推理和评估：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
OUTPUT_ROOT=outputs/affostruction_evaluation/test \
OVERWRITE=false \
SAVE_VISUALS=false \
./run_affostruction_evaluation.sh
```

若前三种消融已完成、仅 `ss_slat` 中断，只补齐 `ss_slat`，随后自动评估全部四种消融：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
OUTPUT_ROOT=outputs/affostruction_evaluation/test \
INFERENCE_METHODS=ss_slat \
RUN_EVALUATION=true \
OVERWRITE=false \
SAVE_VISUALS=false \
./run_affostruction_evaluation.sh
```

只检查缺失样本和磁盘容量，不启动 GPU 推理：

```bash
INFERENCE_METHODS=ss_slat \
OUTPUT_ROOT=outputs/affostruction_evaluation/test \
PREFLIGHT_ONLY=true \
./run_affostruction_evaluation.sh
```

只恢复推理、不立即评估时设置 `RUN_EVALUATION=false`。后续可用
`INFERENCE_METHODS=none RUN_EVALUATION=true` 单独启动评估。完成的 UID 会通过
状态、方法、UID、四个资产文件和 occupancy 载荷共同校验并跳过；残缺目录只清理
已知生成资产后重算。默认保留 2 GiB 磁盘安全空间，大文件先写临时文件，完成后
原子发布。
- 每 4 步一次 decoded-asset supervision
- gsplat RGB/alpha/depth 渲染
- 跨视图 feature/RGB 一致性
- GeomLoss Sinkhorn 几何锁定

输出：

```text
outputs/affostruction_slat/geovis_slat_adapter_last.pt
outputs/affostruction_slat/geovis_slat_adapter_best.pt
outputs/affostruction_slat/train_geovis_slat.jsonl
outputs/affostruction_slat/slat_velocity_debug.npz
outputs/affostruction_slat/slat_visibility_debug.npz
```

监控：

```bash
tail -F outputs/affostruction_slat/train_geovis_slat.jsonl
```

## 9. SLat 断点续训

由于配置采用 `steps_are_total=true`，`--steps 1000` 表示最终目标仍为第 1000 步：

```bash
nohup bash -c '
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
./train_ddp_affostruction.sh slat \
  --output_dir outputs/affostruction_slat \
  --resume outputs/affostruction_slat/geovis_slat_adapter_last.pt \
  --steps 1000 \
  --steps_are_total true
' >> outputs/affostruction_slat.log 2>&1 &

echo "SLat resume started in background. PID: $!"
```

## 10. 一条命令顺序训练 SS 和 SLat

确认两个默认输出目录中没有需要保留的同名运行后：

```bash
nohup bash -c '
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
./train_ddp_affostruction.sh all
' > outputs/affostruction_full_training.log 2>&1 &

echo "Full sequential training (SS→SLat) started in background. PID: $!"
```

更推荐分别执行 SS 和 SLat 命令，以便在两个阶段之间检查 SS 验证结果。

## 11. 最终检查点验收

```bash
python - <<'PY'
import torch

ss_path = "outputs/affostruction_ss_fp32/stochastic_voxel_ss_flow_last.pt"
slat_path = "outputs/affostruction_slat_fp32/geovis_slat_adapter_last.pt"

ss = torch.load(ss_path, map_location="cpu")
slat = torch.load(slat_path, map_location="cpu")

assert ss["architecture_version"] == "affostruction_direct_ss_cross_attention_v1"
assert slat["architecture_version"] == "affostruction_symmetric_slat_cross_attention_v1"
assert int(ss["step"]) == 10000
assert int(slat["step"]) == 1000

print("SS checkpoint:", ss["architecture_version"], ss["step"])
print("SLat checkpoint:", slat["architecture_version"], slat["step"])
print("CHECKPOINT VALIDATION PASSED")
PY
```

正式训练前唯一必须做的外部操作是：等待或停止当前占用 GPU 0/2 的任务。其余代码、配置、数据 manifest 和启动脚本已经就绪。
