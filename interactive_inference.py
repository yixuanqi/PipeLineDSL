"""
交互式推理脚本 - 使用 LoRA 模型
用户可以输入自然语言指令，模型生成 Pipeline JSON
"""

import os
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread


def load_model():
    """加载模型"""
    print("=" * 60)
    print("加载 LoRA 模型...")
    print("=" * 60)
    
    base_model_path = "LLM/models/qwen2.5-3b"
    lora_path = "output/qwen-pipeline-lora"
    
    if not os.path.exists(base_model_path):
        print(f"错误: 基础模型路径不存在: {base_model_path}")
        return None, None
    
    if not os.path.exists(lora_path):
        print(f"错误: LoRA 模型路径不存在: {lora_path}")
        print("请先运行训练脚本: python run_train.py")
        return None, None
    
    print(f"基础模型: {base_model_path}")
    print(f"LoRA 模型: {lora_path}")
    print()
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    print("✅ 模型加载成功!")
    print(f"设备: {model.device}")
    print()
    
    return model, tokenizer


def generate_pipeline_streaming(model, tokenizer, instruction: str, max_new_tokens: int = 1024):
    """流式生成 Pipeline JSON，实时显示生成过程"""
    
    prompt = f"""<|im_start|>system
你是一个数据处理Pipeline生成助手。根据用户的自然语言描述，生成对应的Pipeline JSON配置。
<|im_end|>
<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer
    }
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    print("\n🤖 模型正在思考并生成回答...\n")
    print("=" * 60)
    print("实时生成过程:")
    print("=" * 60)
    
    full_response = ""
    start_time = time.time()
    
    for new_text in streamer:
        full_response += new_text
        print(new_text, end='', flush=True)
    
    thread.join()
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ 生成完成！耗时: {elapsed_time:.2f}秒")
    print("=" * 60)
    
    if "<|im_end|>" in full_response:
        full_response = full_response.split("<|im_end|>")[0]
    
    return full_response.strip()


def generate_pipeline(model, tokenizer, instruction: str, max_new_tokens: int = 1024):
    """生成 Pipeline JSON（非流式，用于批量处理）"""
    
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
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    
    return response.strip()


def parse_json(response: str):
    """尝试解析 JSON"""
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return None


def interactive_inference():
    """交互式推理"""
    model, tokenizer = load_model()
    if model is None:
        return
    
    print("=" * 60)
    print("交互式推理模式")
    print("=" * 60)
    print("输入自然语言指令，模型将生成 Pipeline JSON")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    print()
    
    while True:
        try:
            instruction = input("请输入指令: ").strip()
            
            if not instruction:
                continue
            
            if instruction.lower() in ['quit', 'exit', 'q']:
                print("退出推理模式")
                break
            
            response = generate_pipeline_streaming(model, tokenizer, instruction)
            
            print("\n" + "=" * 60)
            print("生成结果:")
            print("=" * 60)
            print(response)
            print("=" * 60)
            
            parsed = parse_json(response)
            if parsed:
                print("\n解析后的 JSON:")
                print(json.dumps(parsed, ensure_ascii=False, indent=2))
            else:
                print("\n⚠️  警告: 无法解析为有效 JSON")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n退出推理模式")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            print("请重试或输入 'quit' 退出")


def batch_inference(instructions: list, output_file: str = "batch_results.json"):
    """批量推理"""
    model, tokenizer = load_model()
    if model is None:
        return
    
    print(f"\n批量处理 {len(instructions)} 条指令...")
    print()
    
    results = []
    
    for i, instruction in enumerate(instructions, 1):
        print(f"[{i}/{len(instructions)}] {instruction[:50]}...")
        
        response = generate_pipeline(model, tokenizer, instruction)
        parsed = parse_json(response)
        
        results.append({
            "instruction": instruction,
            "response": response,
            "parsed_pipeline": parsed
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 批量处理完成! 结果保存至: {output_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            batch_file = sys.argv[2] if len(sys.argv) > 2 else "batch_instructions.txt"
            
            if os.path.exists(batch_file):
                with open(batch_file, 'r', encoding='utf-8') as f:
                    instructions = [line.strip() for line in f if line.strip()]
                batch_inference(instructions)
            else:
                print(f"错误: 批量指令文件不存在: {batch_file}")
        else:
            print("使用方法:")
            print("  python interactive_inference.py              # 交互式推理")
            print("  python interactive_inference.py --batch [文件名]  # 批量推理")
    else:
        interactive_inference()
