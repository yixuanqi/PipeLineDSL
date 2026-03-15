# Qwen Pipeline DSL 生成模型 - 部署指南

本项目使用 QLoRA 微调 Qwen2.5-3B-Instruct 模型，实现自然语言到 Pipeline DSL 的转换。

## 项目结构

```
dataprocessAssist/
├── train.py                    # 训练脚本
├── inference.py                 # 推理脚本
├── interactive_inference.py       # 交互式推理脚本
├── run_train.py                # 训练启动脚本
├── LLM/
│   └── models/
│       └── qwen2.5-3b/      # 基础模型 (6.17 GB)
├── output/
│   └── qwen-pipeline-lora/     # LoRA 适配器 (57 MB)
└── user_instructions_output/     # 训练数据
```

## 环境要求

### Python 环境

```powershell
# 创建虚拟环境
python -m venv qlora_env

# 激活环境 (Windows)
.\qlora_env\Scripts\Activate.ps1

# 激活环境 (Linux/Mac)
source qlora_env/bin/activate
```

### 依赖安装

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate peft trl sentencepiece bitsandbytes scipy
```

## 快速开始

### 1. 下载基础模型

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir LLM/models/qwen2.5-3b
```

### 2. 训练模型

```powershell
.\qlora_env\Scripts\Activate.ps1
python run_train.py
```

训练参数：
- Epochs: 3
- Batch Size: 2
- Gradient Accumulation: 8
- Learning Rate: 2e-4
- LoRA r: 16, alpha: 32
- 量化: 4-bit NF4

### 3. 推理测试

#### 交互式推理（推荐）

```powershell
.\qlora_env\Scripts\Activate.ps1
python interactive_inference.py
```

#### 批量推理

创建 `batch_instructions.txt` 文件，每行一条指令：

```text
去除边缘数据后计算表面粗糙度
数据滤波然后拟合基准面
去除异常值并填充缺失数据
```

然后运行：

```powershell
.\qlora_env\Scripts\Activate.ps1
python interactive_inference.py --batch batch_instructions.txt
```

## 部署到新电脑

### 方式 1：直接使用 LoRA（推荐）

**优点**：快速、灵活、便于更新

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("LLM/models/qwen2.5-3b")
lora_model = PeftModel.from_pretrained(base_model, "output/qwen-pipeline-lora")

# 推理
output = lora_model.generate(...)
```

### 方式 2：合并模型

**优点**：单一模型文件、推理性能更优

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("LLM/models/qwen2.5-3b")
lora_model = PeftModel.from_pretrained(base_model, "output/qwen-pipeline-lora")

# 合并
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("output/qwen-pipeline-merged")
```

## 推送到 GitHub

### 1. 创建 GitHub 仓库

在 GitHub 创建新仓库，获取仓库 URL。

### 2. 关联远程仓库

```powershell
git remote add origin https://github.com/你的用户名/你的仓库名.git
```

### 3. 推送代码

```powershell
git branch -M master
git push -u origin master
```

### 4. 推送模型（可选）

如果需要推送模型文件，先修改 `.gitignore`：

```gitignore
# 移除这些行，允许推送模型文件
# !LLM/models/
# !output/
```

然后推送：

```powershell
git add LLM/models/ output/
git commit -m "Add model files"
git push -u origin master
```

## 注意事项

1. **模型文件大小**：
   - 基础模型：约 6.17 GB
   - LoRA 适配器：约 57 MB
   - 合并后模型：约 6.2 GB

2. **Git LFS**：
   - 如果推送大模型文件，建议使用 Git LFS
   - 安装：`git lfs install`
   - 追踪：`git lfs track "*.safetensors"`

3. **环境配置**：
   - 确保 Python 3.13+
   - 确保 CUDA 12.4+
   - 确保 8GB+ 显存

4. **数据文件**：
   - 训练数据：`user_instructions_output/`
   - 格式：每个 JSON 包含 `pipeline` 和 `instructions`

## 常见问题

### Q: 如何继续训练？

A: 修改 `run_train.py`，添加 `--resume_from_checkpoint` 参数：

```python
"--resume_from_checkpoint", "output/qwen-pipeline-lora/checkpoint-2250"
```

### Q: 如何调整训练参数？

A: 修改 `run_train.py` 中的参数：

```python
"--num_train_epochs", "5",           # 增加 epoch
"--per_device_train_batch_size", "4",  # 调整 batch size
"--learning_rate", "1e-4",            # 调整学习率
```

### Q: 推理速度慢？

A: 尝试以下优化：
1. 使用 `torch.float16` 而非 `torch.float32`
2. 减少 `max_new_tokens`
3. 使用量化模型（4-bit）

## 技术支持

- **模型**：Qwen2.5-3B-Instruct
- **训练方法**：QLoRA (4-bit 量化)
- **框架**：Transformers + PEFT + TRL
- **训练数据**：12,000 样本
- **最终准确率**：99%

## 许可证

本项目遵循 Apache 2.0 许可证。
