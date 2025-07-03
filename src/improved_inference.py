#!/usr/bin/env python3
"""
Improved Sinhala LLM Inference Script
Enhanced chat capabilities with conversation context management
"""

import torch
import sentencepiece as spm
import argparse
import os
from improved_model_architecture import ImprovedSinhalaLLM, ImprovedModelConfig
from typing import List, Dict, Optional
import re

class ImprovedSinhalaLLMInference:
    """Enhanced inference engine with better chat capabilities"""
    
    def __init__(self, model_path='../models/improved_sinhala_llm_best.pt', device=None):
        """
        Initialize the improved inference engine
        
        Args:
            model_path (str): Path to the trained model file
            device (str): Device to run inference on
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model checkpoint
        print(f"Loading improved model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Reconstruct config
        config_dict = checkpoint['config']
        
        # Filter config for model initialization
        model_config_keys = {
            'vocab_size', 'hidden_size', 'num_layers', 'num_attention_heads', 
            'intermediate_size', 'max_position_embeddings', 'dropout', 
            'attention_dropout', 'layer_norm_eps', 'initializer_range', 
            'use_cache', 'pad_token_id', 'bos_token_id', 'eos_token_id',
            'user_token_id', 'assistant_token_id', 'end_turn_token_id',
            'use_rope', 'use_flash_attention'
        }
        
        filtered_config = {k: v for k, v in config_dict.items() if k in model_config_keys}
        self.config = ImprovedModelConfig(**filtered_config)
        
        # Load tokenizer
        print("Loading tokenizer...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(script_dir, '..', 'models', 'tokenizer', 'sinhala_sp.model')
        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        
        # Initialize and load model
        print("Initializing improved model...")
        self.model = ImprovedSinhalaLLM(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Conversation history
        self.conversation_history = []
        self.max_history_turns = 5
        
        # Special tokens
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.end_token = "<|end|>"
        self.system_token = "<|system|>"
        
        print(f"Model loaded successfully! Parameters: {self.count_parameters():,}")
        print(f"Training epoch: {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.model.parameters())
    
    def add_to_history(self, user_input: str, assistant_response: str):
        """Add conversation turn to history"""
        self.conversation_history.append({
            'user': user_input,
            'assistant': assistant_response
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.")
    
    def build_conversation_context(self, current_input: str, include_history: bool = True) -> str:
        """Build conversation context with history"""
        context_parts = []
        
        # Add conversation history
        if include_history and self.conversation_history:
            for turn in self.conversation_history:
                context_parts.append(f"{self.user_token}\n{turn['user']}")
                context_parts.append(f"{self.assistant_token}\n{turn['assistant']}\n{self.end_token}")
        
        # Add current user input
        context_parts.append(f"{self.user_token}\n{current_input}")
        context_parts.append(f"{self.assistant_token}")
        
        return "\n".join(context_parts)
    
    def generate_response(
        self, 
        user_input: str, 
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        include_history: bool = True,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a conversational response
        
        Args:
            user_input (str): User's input message
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling
            top_k (int): Top-k sampling
            repetition_penalty (float): Repetition penalty
            include_history (bool): Whether to include conversation history
            system_prompt (str): Optional system prompt
        
        Returns:
            str: Generated response
        """
        # Build conversation context
        conversation_context = self.build_conversation_context(user_input, include_history)
        
        # Add system prompt if provided
        if system_prompt:
            conversation_context = f"{self.system_token}\n{system_prompt}\n" + conversation_context
        
        print(f"Input context length: {len(conversation_context)} characters")
        
        # Tokenize
        tokens = self.tokenizer.encode(conversation_context, out_type=int)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated = self.model.generate_chat_response(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                eos_token_id=self.config.end_turn_token_id
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        
        # Extract only the assistant's response
        assistant_response = self.extract_assistant_response(generated_text, conversation_context)
        
        # Add to conversation history
        self.add_to_history(user_input, assistant_response)
        
        return assistant_response
    
    def extract_assistant_response(self, generated_text: str, original_context: str) -> str:
        """Extract the assistant's response from generated text"""
        # Find the last assistant token in the generated text
        if f"{self.assistant_token}" in generated_text:
            # Split by assistant token and get the last part
            parts = generated_text.split(f"{self.assistant_token}")
            if len(parts) > 1:
                response = parts[-1].strip()
                
                # Remove end token if present
                if self.end_token in response:
                    response = response.split(self.end_token)[0].strip()
                
                # Remove any remaining special tokens
                response = self.clean_response(response)
                
                return response
        
        # Fallback: return generated text after original context
        if len(generated_text) > len(original_context):
            response = generated_text[len(original_context):].strip()
            return self.clean_response(response)
        
        return "කම්සෝධා මා නොහිතින් කියන්න එපා... (Sorry, I couldn't generate a proper response)"
    
    def clean_response(self, response: str) -> str:
        """Clean up the response text"""
        # Remove special tokens
        special_tokens = [self.user_token, self.assistant_token, self.end_token, self.system_token]
        for token in special_tokens:
            response = response.replace(token, "")
        
        # Remove extra whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Remove incomplete sentences at the end
        if response and not response[-1] in '.!?':
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '.'.join(sentences[:-1]) + '.'
        
        return response
    
    def interactive_chat(self):
        """Enhanced interactive chat mode"""
        print("\n" + "="*60)
        print("🇱🇰 Improved Sinhala LLM Chat Mode 🇱🇰")
        print("Type 'quit' or 'exit' to end")
        print("Type 'clear' to clear conversation history")
        print("Type 'help' for more commands")
        print("="*60 + "\n")
        
        # Optional system prompt
        system_prompt = input("System prompt (optional, press Enter to skip): ").strip()
        if system_prompt:
            print(f"System prompt set: {system_prompt}\n")
        else:
            system_prompt = None
        
        while True:
            try:
                user_input = input("🧑 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! / ගිහින් එන්න!")
                    break
                
                if user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower().startswith('temp '):
                    # Change temperature
                    try:
                        new_temp = float(user_input.split()[1])
                        self.current_temperature = new_temp
                        print(f"Temperature set to {new_temp}")
                        continue
                    except:
                        print("Invalid temperature. Use: temp <number>")
                        continue
                
                if not user_input:
                    continue
                
                print("🤖 Generating response...")
                
                # Generate response
                response = self.generate_response(
                    user_input,
                    max_new_tokens=400,
                    temperature=getattr(self, 'current_temperature', 0.7),
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    include_history=True,
                    system_prompt=system_prompt
                )
                
                print(f"🤖 Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! / ගිහින් එන්න!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again.")
    
    def show_help(self):
        """Show help commands"""
        print("\n📋 Available Commands:")
        print("  quit/exit/q    - Exit chat")
        print("  clear          - Clear conversation history")
        print("  temp <number>  - Set temperature (0.1-2.0)")
        print("  help           - Show this help")
        print("  <message>      - Send a message\n")
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts"""
        responses = []
        for prompt in prompts:
            response = self.generate_response(prompt, include_history=False, **kwargs)
            responses.append(response)
        return responses
    
    def evaluate_conversation_quality(self, test_conversations: List[Dict]) -> Dict:
        """Evaluate conversation quality metrics"""
        metrics = {
            'avg_response_length': 0,
            'coherence_score': 0,
            'relevance_score': 0
        }
        
        total_length = 0
        for conv in test_conversations:
            user_input = conv['user']
            expected_response = conv.get('expected', '')
            
            generated_response = self.generate_response(user_input, include_history=False)
            total_length += len(generated_response.split())
        
        metrics['avg_response_length'] = total_length / len(test_conversations) if test_conversations else 0
        
        return metrics

def main():
    """Main function with enhanced CLI"""
    parser = argparse.ArgumentParser(description='Improved Sinhala LLM for enhanced conversations')
    parser.add_argument('--model', default='../models/improved_sinhala_llm_best.pt',
                        help='Path to improved model file')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                        help='Device to use for inference')
    parser.add_argument('--prompt', type=str,
                        help='Single prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=400,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--chat', action='store_true',
                        help='Start interactive chat mode')
    parser.add_argument('--system-prompt', type=str,
                        help='System prompt for conversation context')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        llm = ImprovedSinhalaLLMInference(model_path=args.model, device=args.device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    if args.chat:
        # Start enhanced chat mode
        llm.interactive_chat()
    elif args.prompt:
        # Single prompt generation
        print("🤖 Generating response...")
        response = llm.generate_response(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            include_history=False,
            system_prompt=args.system_prompt
        )
        print(f"\n🤖 Response: {response}")
    else:
        print("Please specify either --chat for interactive mode or --prompt for single generation")
        print("Use --help for more options")

if __name__ == "__main__":
    main() 