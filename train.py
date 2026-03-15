"""
QLoRA 训练脚本 - Qwen2.5-3B-Instruct
将自然语言转换为 Pipeline DSL
"""

import os
import json
import glob
import torch
from typing import List, Dict, Any
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer
from datasets import Dataset
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="LLM/models/qwen2.5-3b",
        metadata={"help": "模型路径"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "是否使用4-bit量化"}
    )
    use_nested_quant: bool = field(
        default=True,
        metadata={"help": "是否使用嵌套量化"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "计算数据类型"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "量化类型"}
    )


@dataclass
class LoRAArguments:
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA秩"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "目标模块"}
    )


@dataclass
class DataArguments:
    data_dir: str = field(
        default="user_instructions_output",
        metadata={"help": "数据目录"}
    )
    max_input_length: int = field(
        default=512,
        metadata={"help": "最大输入长度"}
    )
    max_output_length: int = field(
        default=1024,
        metadata={"help": "最大输出长度"}
    )
    max_seq_length: int = field(
        default=1536,
        metadata={"help": "最大序列长度"}
    )


class PipelineDataLoader:
    """Pipeline 数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def load_all_instructions(self) -> List[Dict[str, Any]]:
        """加载所有 instruction json 文件"""
        all_data = []
        
        instruction_files = sorted(glob.glob(
            os.path.join(self.data_dir, "instructions_part*.json")
        ))
        
        print(f"找到 {len(instruction_files)} 个指令文件")
        
        for file_path in instruction_files:
            print(f"加载: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        
        print(f"总共加载 {len(all_data)} 条原始数据")
        return all_data
    
    def convert_to_training_format(self, raw_data: List[Dict]) -> List[Dict]:
        """将原始数据转换为训练格式"""
        training_data = []
        
        for item in raw_data:
            pipeline = item.get("pipeline", {})
            instructions = item.get("instructions", [])
            
            output_json = json.dumps(pipeline, ensure_ascii=False, indent=2)
            
            for instruction in instructions:
                training_data.append({
                    "instruction": instruction,
                    "output": output_json
                })
        
        print(f"转换后训练样本数: {len(training_data)}")
        return training_data
    
    def create_dataset(self) -> Dataset:
        """创建 HuggingFace Dataset"""
        raw_data = self.load_all_instructions()
        training_data = self.convert_to_training_format(raw_data)
        
        return Dataset.from_list(training_data)


def format_instruction(sample: Dict) -> str:
    """格式化指令为模型输入格式"""
    return f"""<|im_start|>system
你是一个数据处理Pipeline生成助手。根据用户的自然语言描述，生成对应的Pipeline JSON配置。
<|im_end|>
<|im_start|>user
{sample['instruction']}
<|im_end|>
<|im_start|>assistant
{sample['output']}
<|im_end|>"""


def load_model_and_tokenizer(model_args: ModelArguments):
    """加载模型和tokenizer"""
    
    print(f"加载模型: {model_args.model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_args.use_4bit:
        compute_dtype = getattr(torch, model_args.bnb_4bit_compute_dtype)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_nested_quant,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    return model, tokenizer


def prepare_model_for_training(model, lora_args: LoRAArguments):
    """准备模型用于训练"""
    
    model = prepare_model_for_kbit_training(model)
    
    target_modules = lora_args.target_modules.split(",")
    
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


class SavePeftModelCallback(transformers.TrainerCallback):
    """保存 PEFT 模型的回调"""
    
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
        
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
    
    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)


def main():
    parser = transformers.HfArgumentParser((
        ModelArguments, 
        LoRAArguments, 
        DataArguments, 
        TrainingArguments
    ))
    model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("=" * 60)
    print("QLoRA 训练配置")
    print("=" * 60)
    print(f"模型路径: {model_args.model_name_or_path}")
    print(f"数据目录: {data_args.data_dir}")
    print(f"输出目录: {training_args.output_dir}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Batch Size: {training_args.per_device_train_batch_size}")
    print(f"Gradient Accumulation: {training_args.gradient_accumulation_steps}")
    print(f"LoRA r: {lora_args.lora_r}, alpha: {lora_args.lora_alpha}")
    print("=" * 60)
    
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    model = prepare_model_for_training(model, lora_args)
    
    if training_args.resume_from_checkpoint:
        checkpoint_path = training_args.resume_from_checkpoint
        adapter_path = os.path.join(checkpoint_path, "adapter_model")
        if os.path.exists(adapter_path):
            print(f"\n从 checkpoint 加载 adapter 权重: {adapter_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            print("Adapter 权重加载成功!")
    
    data_loader = PipelineDataLoader(data_args.data_dir)
    dataset = data_loader.create_dataset()
    
    print(f"\n数据集样本数: {len(dataset)}")
    print(f"样本示例:\n{format_instruction(dataset[0])[:500]}...")
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=format_instruction,
        args=training_args,
        callbacks=[SavePeftModelCallback()],
    )
    
    print("\n开始训练...")
    resume_checkpoint = None
    if training_args.resume_from_checkpoint:
        state_path = os.path.join(training_args.resume_from_checkpoint, "trainer_state.json")
        if os.path.exists(state_path):
            resume_checkpoint = training_args.resume_from_checkpoint
            print(f"从 checkpoint 恢复训练状态: {resume_checkpoint}")
        else:
            print("Checkpoint 缺少 trainer_state.json，将从头开始训练（但已加载 adapter 权重）")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    print("\n保存最终模型...")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    print(f"\n训练完成! 模型保存至: {training_args.output_dir}")


if __name__ == "__main__":
    main()
