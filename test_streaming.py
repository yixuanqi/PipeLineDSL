"""
测试流式输出功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from interactive_inference import load_model, generate_pipeline_streaming, parse_json
import json

def test_streaming():
    """测试流式输出"""
    print("=" * 60)
    print("测试流式输出功能")
    print("=" * 60)
    
    model, tokenizer = load_model()
    if model is None:
        return
    
    test_instruction = "分析表面粗糙度"
    
    print(f"\n测试指令: {test_instruction}")
    print()
    
    response = generate_pipeline_streaming(model, tokenizer, test_instruction)
    
    print("\n" + "=" * 60)
    print("最终结果:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    
    parsed = parse_json(response)
    if parsed:
        print("\n解析后的 JSON:")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    else:
        print("\n⚠️  警告: 无法解析为有效 JSON")

if __name__ == "__main__":
    test_streaming()
