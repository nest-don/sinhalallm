#!/usr/bin/env python3
"""
Quick Demo: Using the Sinhala LLM
Simple script to quickly test the trained model
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from use_model import SinhalaLLMInference

def main():
    """Quick demonstration of the Sinhala LLM"""
    print("🇱🇰 Sinhala LLM Quick Demo")
    print("=" * 40)
    
    # Load the model
    try:
        print("Loading the Sinhala LLM model...")
        llm = SinhalaLLMInference(model_path='../models/sinhala_llm_best.pt')
        print("✅ Model loaded successfully!\n")
        
        # Test with some example prompts
        test_prompts = [
            "ශ්‍රී ලංකාව",
            "කොළඹ නගරය",
            "අධ්‍යාපනය වැදගත් වන්නේ",
            "සිංහල භාෂාව ලස්සන",
            "මේ රටේ ඉතිහාසය"
        ]
        
        print("🔮 Generating text samples...")
        print("-" * 40)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"{i}. Input: {prompt}")
            generated = llm.complete_text(prompt, max_length=60)
            print(f"   Output: {generated}")
            print()
        
        print("✨ Demo completed! Try 'python use_model.py --chat' for interactive mode.")
        
    except FileNotFoundError:
        print("❌ Model file not found! Make sure 'sinhala_llm_best.pt' exists in models/ directory.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 