#!/usr/bin/env python3
"""
Sinhala LLM Inference Script
Load and use the trained Sinhala language model for text generation
"""

import torch
import sentencepiece as spm
import argparse
import os
from model_architecture import SinhalaLLM, ModelConfig

class SinhalaLLMInference:
    """Class for loading and using the trained Sinhala LLM"""
    
    def __init__(self, model_path='sinhala_llm_best.pt', device=None):
        """
        Initialize the inference engine
        
        Args:
            model_path (str): Path to the trained model file
            device (str): Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Reconstruct config from checkpoint
        config_dict = checkpoint['config']
        
        # Filter out training-specific parameters that aren't part of ModelConfig
        model_config_keys = {
            'vocab_size', 'hidden_size', 'num_layers', 'num_attention_heads', 
            'intermediate_size', 'max_position_embeddings', 'dropout', 
            'layer_norm_eps', 'initializer_range', 'use_cache', 
            'pad_token_id', 'bos_token_id', 'eos_token_id'
        }
        
        filtered_config = {k: v for k, v in config_dict.items() if k in model_config_keys}
        self.config = ModelConfig(**filtered_config)
        
        # Load tokenizer
        print("Loading tokenizer...")
        # Get the directory of this script file and build path to tokenizer
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(script_dir, '..', 'models', 'tokenizer', 'sinhala_sp.model')
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        
        # Initialize and load model
        print("Initializing model...")
        self.model = SinhalaLLM(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully! Parameters: {self.count_parameters():,}")
        print(f"Training epoch: {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def generate_text(self, prompt, max_length=100, temperature=0.8, top_p=0.9, do_sample=True):
        """
        Generate text from a prompt
        
        Args:
            prompt (str): Input text prompt in Sinhala
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature (0.1-2.0)
            top_p (float): Top-p sampling parameter (0.1-1.0)
            do_sample (bool): Whether to use sampling or greedy decoding
        
        Returns:
            str: Generated text
        """
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt, out_type=int)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        print(f"Input prompt: {prompt}")
        print(f"Input tokens: {len(tokens)}")
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.config.pad_token_id
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        
        return generated_text
    
    def chat_mode(self):
        """Interactive chat mode"""
        print("\n" + "="*50)
        print("Sinhala LLM Chat Mode")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'help' for commands")
        print("="*50 + "\n")
        
        while True:
            try:
                prompt = input("You: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if prompt.lower() == 'help':
                    print("\nCommands:")
                    print("- help: Show this help message")
                    print("- quit/exit/q: Exit chat mode")
                    print("- Just type any Sinhala text to generate a response")
                    print()
                    continue
                
                if not prompt:
                    continue
                
                # Add user prefix for conversation format
                formatted_prompt = f"<|user|>{prompt}<|assistant|>"
                
                print("Generating response...")
                response = self.generate_text(
                    formatted_prompt,
                    max_length=200,
                    temperature=0.8,
                    top_p=0.9
                )
                
                # Extract assistant response
                if "<|assistant|>" in response:
                    assistant_response = response.split("<|assistant|>")[-1]
                    if "<|end|>" in assistant_response:
                        assistant_response = assistant_response.split("<|end|>")[0]
                    print(f"Assistant: {assistant_response.strip()}")
                else:
                    print(f"Assistant: {response}")
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.")
    
    def complete_text(self, prompt, max_length=100):
        """
        Complete a text prompt (simple completion without conversation format)
        
        Args:
            prompt (str): Text to complete
            max_length (int): Maximum length of completion
        
        Returns:
            str: Completed text
        """
        return self.generate_text(prompt, max_length=max_length, temperature=0.7, do_sample=True)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Use trained Sinhala LLM for text generation')
    parser.add_argument('--model', default='../models/sinhala_llm_best.pt', 
                        help='Path to model file (default: ../models/sinhala_llm_best.pt)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='Device to use for inference')
    parser.add_argument('--prompt', type=str, 
                        help='Text prompt for generation')
    parser.add_argument('--max-length', type=int, default=100,
                        help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--chat', action='store_true',
                        help='Start interactive chat mode')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        llm = SinhalaLLMInference(model_path=args.model, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if args.chat:
        # Start chat mode
        llm.chat_mode()
    elif args.prompt:
        # Generate from prompt
        print(f"Generating text from prompt: {args.prompt}")
        generated = llm.generate_text(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"\nGenerated text:\n{generated}")
    else:
        # Default examples
        print("Running example generations...")
        
        examples = [
            "ශ්‍රී ලංකාව",
            "කොළඹ නගරය",
            "අධ්‍යාපනය වැදගත් වන්නේ",
            "තාක්ෂණය අනාගතයට",
            "සිංහල භාෂාව"
        ]
        
        for prompt in examples:
            print(f"\nPrompt: {prompt}")
            generated = llm.complete_text(prompt, max_length=80)
            print(f"Generated: {generated}")
            print("-" * 50)

if __name__ == "__main__":
    main() 