"""
训练启动脚本
针对 8GB 显存优化的 QLoRA 训练配置
"""

import subprocess
import sys
import os

def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    cmd = [
        sys.executable, "train.py",
        
        "--model_name_or_path", "LLM/models/qwen2.5-3b",
        "--use_4bit", "True",
        "--use_nested_quant", "True",
        "--bnb_4bit_compute_dtype", "bfloat16",
        "--bnb_4bit_quant_type", "nf4",
        
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--lora_dropout", "0.05",
        "--target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        
        "--data_dir", "user_instructions_output",
        "--max_input_length", "512",
        "--max_output_length", "1024",
        
        "--output_dir", "output/qwen-pipeline-lora",
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "8",
        "--learning_rate", "2e-4",
        "--weight_decay", "0.01",
        "--warmup_ratio", "0.1",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "10",
        "--save_steps", "100",
        "--save_total_limit", "3",
        "--bf16", "True",
        "--gradient_checkpointing", "True",
        "--optim", "paged_adamw_8bit",
        "--max_grad_norm", "1.0",
        "--report_to", "none",
        "--dataloader_num_workers", "0",
        "--remove_unused_columns", "False",
        "--max_seq_length", "1536",
        "--resume_from_checkpoint", "output/qwen-pipeline-lora/checkpoint-100",
    ]
    
    print("=" * 60)
    print("启动 QLoRA 训练")
    print("=" * 60)
    print("命令:")
    print(" ".join(cmd))
    print("=" * 60)
    
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
