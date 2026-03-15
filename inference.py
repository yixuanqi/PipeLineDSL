"""
推理脚本 - 使用训练好的 LoRA 模型进行推理
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model_path: str, lora_path: str):
    """加载基础模型和 LoRA 适配器"""
    
    print(f"加载基础模型: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"加载 LoRA 适配器: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    return model, tokenizer


def generate_pipeline(model, tokenizer, instruction: str, max_new_tokens: int = 1024):
    """根据自然语言指令生成 Pipeline JSON"""
    
    prompt = f"""<|im_start|>system
你是一个数据处理Pipeline生成助手。根据用户的自然语言描述，生成对应的Pipeline JSON配置。
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    
    return response.strip()


def parse_pipeline_json(response: str):
    """尝试解析生成的 JSON"""
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return None


def main():
    BASE_MODEL_PATH = "LLM/models/qwen2.5-3b"
    LORA_PATH = "output/qwen-pipeline-lora"
    
    if not os.path.exists(LORA_PATH):
        print(f"错误: LoRA 模型路径不存在: {LORA_PATH}")
        print("请先运行训练脚本: python run_train.py")
        return
    
    model, tokenizer = load_model(BASE_MODEL_PATH, LORA_PATH)
    
    test_instructions = [
        "帮我去除边缘数据，然后平滑处理，最后计算表面粗糙度",
        "数据里有些坏点，清理掉之后给我算个表面粗糙度",
        "请对数据进行滤波处理，然后拟合基准面",
        "去除异常值后进行数据填充，最后分析结果",
    ]
    
    print("\n" + "=" * 60)
    print("推理测试")
    print("=" * 60)
    
    for instruction in test_instructions:
        print(f"\n指令: {instruction}")
        print("-" * 40)
        
        response = generate_pipeline(model, tokenizer, instruction)
        print(f"生成结果:\n{response}")
        
        parsed = parse_pipeline_json(response)
        if parsed:
            print(f"\n解析后的 Pipeline:\n{json.dumps(parsed, ensure_ascii=False, indent=2)}")
        else:
            print("\n警告: 无法解析为有效 JSON")
        
        print("=" * 60)


if __name__ == "__main__":
    main()
