# GUI-Owl-1.5-2B-Instruct 模型转换实现总结

## 概述

本文档总结了为 ExecuTorch 项目添加 mPLUG/GUI-Owl-1.5-2B-Instruct 模型转换支持所做的所有更改。

## 模型信息

- **模型名称**: GUI-Owl-1.5-2B-Instruct
- **HuggingFace ID**: `mPLUG/GUI-Owl-1.5-2B-Instruct`
- **基础架构**: Qwen3VL (Qwen3 Vision Language)
- **参数量**: 2B
- **特点**: 动态分辨率支持，专为 GUI 理解任务设计

## 添加的文件

### 1. 模型定义目录 `examples/models/gui_owl/`

#### `__init__.py`
```python
# 模型注册入口
from executorch.examples.models.gui_owl.convert_weights import convert_weights
from executorch.examples.models.llama.model import Llama2Model

class Qwen3VLModel(Llama2Model):
    """Qwen3VL 架构的模型基类"""
```

#### `convert_weights.py`
```python
# 权重转换脚本
# - 将 HuggingFace 格式的 Qwen3VL 权重转换为 Meta 格式
# - 提取 text decoder 权重
# - 处理绑定的 embeddings
```

#### `export_gui_owl.py`
```python
# 导出脚本（模板）
# - 定义 GUIOwlEdgeManager
# - 导出 vision encoder
# - 导出 text decoder
# - 导出 token embedding
```

#### `config/2b_config.json`
```json
{
  "dim": 2048,
  "n_heads": 16,
  "n_layers": 28,
  "vocab_size": 151936,
  "rope_theta": 5000000,
  "use_hf_rope": true,
  ...
}
```

#### `README.md`
模型使用说明和文档

---

## 修改的文件

### 1. `examples/qualcomm/oss_scripts/llama/model/vision_encoder.py`

添加了 `Qwen3VLVisionEncoder` 类：

```python
class Qwen3VLVisionEncoder(torch.nn.Module):
    """
    Qwen3VL 视觉编码器，支持动态分辨率

    特性:
    - Conv3d patch embedding
    - Rotary position embeddings
    - Spatial merge for downsampling
    """
```

**修改内容**:
- 导入 Qwen3VL 相关类
- 实现 Qwen3VLVisionEncoder 类
- 支持 512x512 默认输入分辨率
- spatial_merge_size=2 用于序列长度缩减

---

### 2. `examples/qualcomm/oss_scripts/llama/encoder/encoder_config.py`

添加了 `GUIOwlEncoder` 配置：

```python
@dataclass(init=False, frozen=True)
class GUIOwlEncoder(VisionModalityConfig):
    encoder_class = Qwen3VLVisionEncoder
    img_seq_len = 256  # (512/16)^2 / 4 = 256
    img_resized_h = 512
    img_resized_w = 512
    img_url = "https://cdn.britannica.com/61/93061-050-99147DCE/..."
    quant_recipe = InternVL3_Encoder_QuantRecipe
```

**修改内容**:
- 导入 Qwen3VLVisionEncoder
- 定义 GUIOwlEncoder 配置类

---

### 3. `examples/qualcomm/oss_scripts/llama/encoder/encoder_quant_recipe.py`

添加了 `GUIOwl_Encoder_QuantRecipe`:

```python
class GUIOwl_Encoder_QuantRecipe(EncoderQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w
    # 使用与 InternVL3 相同的量化配置
```

---

### 4. `examples/qualcomm/oss_scripts/llama/encoder/__init__.py`

**修改内容**:
- 导出 `GUIOwlEncoder`
- 导出 `GUIOwl_Encoder_QuantRecipe`

---

### 5. `examples/qualcomm/oss_scripts/llama/static_llm_quant_recipe.py`

添加了 `GUIOwl_2B_QuantRecipe`:

```python
class GUIOwl_2B_QuantRecipe(StaticLLMQuantRecipe):
    default_quant_dtype = QuantDtype.use_16a8w
    # 适用于 GUI-Owl 文本解码器的量化配置
```

---

### 6. `examples/qualcomm/oss_scripts/llama/__init__.py`

**修改内容**:

1. 导入权重转换函数:
```python
from executorch.examples.models.gui_owl import (
    convert_weights as convert_gui_owl_weights,
)
```

2. 导入量化配置:
```python
from executorch.examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import (
    ...
    GUIOwl_2B_QuantRecipe,
)
```

3. 注册模型配置:
```python
@register_llm_model(
    "gui_owl_2b_instruct",
    vision_encoder=GUIOwlEncoder,
)
@dataclass(init=False, frozen=True)
class GUIOwl_2B_Instruct(LLMModelConfig):
    repo_id: str = "mPLUG/GUI-Owl-1.5-2B-Instruct"
    params_path: str = os.path.join(
        BASE_DIR, "../../../models/gui_owl/2b_config.json"
    )
    convert_weights = convert_gui_owl_weights
    transform_weight = False
    instruct_model = True
    num_sharding = 1
    masked_softmax = True
    quant_recipe = GUIOwl_2B_QuantRecipe
```

---

## 使用方法

### 1. 下载模型

```bash
huggingface-cli download mPLUG/GUI-Owl-1.5-2B-Instruct --local-dir ./gui_owl_weights
```

### 2. 转换权重

```bash
python -m executorch.examples.models.gui_owl.convert_weights \
    ./gui_owl_weights \
    gui_owl_weights_converted.pt
```

### 3. 使用 llama.py 导出（高通设备）

```bash
python examples/qualcomm/oss_scripts/llama/llama.py \
    --decoder_model gui_owl_2b_instruct \
    --checkpoint gui_owl_weights_converted.pt \
    --params examples/models/gui_owl/config/2b_config.json \
    --tokenizer_model ./gui_owl_weights/tokenizer.json \
    --prompt "Describe this image" \
    --image_path ./test.jpg \
    --max_seq_len 768 \
    --artifact ./gui_owl_output
```

### 4. 使用 XNNPACK 导出

```bash
python -m executorch.examples.models.gui_owl.export_gui_owl \
    --max-context-len 768 \
    --max-seq-len 768 \
    --pte-name gui_owl_xnnpack.pte
```

---

## 架构细节

### GUI-Owl-1.5-2B-Instruct 配置

| 组件 | 参数 | 值 |
|------|------|-----|
| **Text Decoder** | layers | 28 |
| | hidden_dim | 2048 |
| | n_heads | 16 |
| | n_kv_heads | 8 |
| | vocab_size | 151936 |
| | rope_theta | 5000000 |
| **Vision Encoder** | depth | 24 |
| | hidden_size | 1024 |
| | patch_size | 16 |
| | spatial_merge_size | 2 |
| | out_hidden_size | 2048 |
| | num_position_embeddings | 2304 |

### 动态分辨率处理

Qwen3VL 通过以下机制支持动态分辨率：

1. **Patch Embedding**: 使用 Conv3d 处理任意尺寸的输入
   - `kernel_size = [temporal_patch_size, patch_size, patch_size]`
   - 支持 2D 图像（temporal_patch_size 用于视频）

2. **Spatial Merge**:
   ```
   输出序列长度 = (H/patch_size) × (W/patch_size) / (spatial_merge_size²)

   对于 512×512 输入:
   = (512/16) × (512/16) / (2×2)
   = 32 × 32 / 4
   = 256
   ```

3. **Rotary Position Embedding**: 适应不同序列长度

---

## 参考资料

- [Llava 模型实现](examples/models/llava/model.py)
- [InternVL3 模型实现](examples/models/internvl3/)
- [Qwen3 模型实现](examples/models/qwen3/)
- [HuggingFace GUI-Owl](https://huggingface.co/mPLUG/GUI-Owl-1.5-2B-Instruct)

---

## 注意事项

1. **导入依赖**: 需要 `transformers >= 4.57.0` 以支持 Qwen3VL
2. **内存要求**: 导出 2B 模型约需 16GB 内存
3. **Token ID**:
   - image_token_id: 151655
   - vision_start_token_id: 151652
   - vision_end_token_id: 151653
4. **Chat 模板**: 作为 instruct 模型，需使用正确的对话格式
