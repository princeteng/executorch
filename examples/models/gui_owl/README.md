# GUI-Owl-1.5-2B-Instruct 模型转换指南

## 模型概述

GUI-Owl-1.5-2B-Instruct 是由 mPLUG 团队开发的视觉语言模型，基于 Qwen3VL 架构。该模型专为 GUI 理解和交互任务设计，支持动态分辨率图像处理。

### 模型架构特点

- **Text Decoder**: 基于 Qwen3 架构，28 层，2048 隐藏维度
- **Vision Encoder**: Qwen3VL Vision Transformer
  - 24 层，1024 隐藏维度
  - Patch size: 16
  - Spatial merge size: 2（支持动态分辨率）
  - 输出投影到 2048 维度
- **动态分辨率**: 通过 spatial_merge 机制支持不同尺寸的图像输入

## 文件结构

```
examples/models/gui_owl/
├── __init__.py              # 模型注册
├── convert_weights.py       # 权重转换脚本
├── export_gui_owl.py        # 导出脚本
└── config/
    └── 2b_config.json       # 模型配置
```

## 使用方法

### 1. 准备模型权重

从 Hugging Face 下载模型权重：

```bash
huggingface-cli download mPLUG/GUI-Owl-1.5-2B-Instruct --local-dir ./gui_owl_weights
```

### 2. 转换权重格式

将 Hugging Face 格式的权重转换为 ExecuTorch 格式：

```bash
python -m executorch.examples.models.gui_owl.convert_weights \
    ./gui_owl_weights \
    gui_owl_weights_converted.pt
```

### 3. 使用 llama.py 进行模型转换

对于高通设备，使用 qualcomm 的 llama.py 脚本：

```bash
python -m executorch.examples.qualcomm.oss_scripts.llama.llama \
    --decoder_model gui_owl_2b_instruct \
    --checkpoint gui_owl_weights_converted.pt \
    --params examples/models/gui_owl/config/2b_config.json \
    --tokenizer_model ./gui_owl_weights/tokenizer.json \
    --prompt "Describe this image:" \
    --image_path ./test_image.jpg \
    --max_seq_len 768 \
    --max_context_len 768 \
    --artifact ./gui_owl_output
```

### 4. 使用 XNNPACK 后端导出（可选）

```bash
python -m executorch.examples.models.gui_owl.export_gui_owl \
    --max-context-len 768 \
    --max-seq-len 768 \
    --pte-name gui_owl_xnnpack.pte \
    --with-artifacts
```

## 模型配置参数

在 `llama/__init__.py` 中的 `GUIOwl_2B_Instruct` 类包含以下配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| repo_id | mPLUG/GUI-Owl-1.5-2B-Instruct | HuggingFace 模型 ID |
| params_path | gui_owl/config/2b_config.json | 模型配置文件 |
| vision_encoder | GUIOwlEncoder | 视觉编码器配置 |
| img_resized_h | 512 | 图像目标高度 |
| img_resized_w | 512 | 图像目标宽度 |
| img_seq_len | 256 | 图像序列长度 |
| transform_weight | False | 是否需要权重转换 |
| instruct_model | True | 是否使用指令模板 |
| num_sharding | 1 | 分片数量 |

## 动态分辨率处理

GUI-Owl 基于 Qwen3VL 架构，通过以下机制支持动态分辨率：

1. **Patch Embedding**: 使用 Conv3d 处理任意尺寸的输入图像
2. **Spatial Merge**: 通过 `spatial_merge_size=2` 将相邻 patch 合并，减少序列长度
3. **位置编码**: 使用旋转位置编码（RoPE），适应不同序列长度

对于高分辨率图像，模型会自动调整：
- 图像首先被调整到目标尺寸（默认 512x512）
- Patch 嵌入后，通过 spatial merge 减少序列长度
- 最终序列长度 = (H/patch_size) × (W/patch_size) / (spatial_merge_size²)

## 注意事项

1. **内存要求**: 导出 2B 模型需要约 16GB 内存
2. **图像预处理**: 输入图像需要按照 Qwen3VL 的预处理方式进行调整和归一化
3. **Token 格式**: GUI-Owl 使用特殊的 image token（ID: 151655）来标记图像位置
4. **Chat 模板**: 作为指令模型，需要使用正确的 chat 模板格式

## 参考实现

- [Llava 模型实现](../llava/model.py) - 参考 VLM 模型结构
- [InternVL3 模型实现](../internvl3/) - 参考 QNN 后端集成
- [Qwen3 模型实现](../qwen3/) - 参考 Qwen 系列权重转换
