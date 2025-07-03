#!/usr/bin/env python3
"""
Improved Sinhala LLM Training Script
Enhanced for better chat performance with conversation-aware training
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer
import sentencepiece as spm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import numpy as np
from pathlib import Path
import random
import math
from typing import List, Dict, Optional

# Import our improved model
from improved_model_architecture import ImprovedSinhalaLLM, ImprovedModelConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/improved_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConversationDataset(Dataset):
    """Enhanced dataset class for conversation training with role awareness"""
    
    def __init__(self, data_file, tokenizer, max_length=2048, tokenizer_type='sentencepiece'):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer_type = tokenizer_type
        
        # Special tokens
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.end_token = "<|end|>"
        self.system_token = "<|system|>"
        
        logger.info(f"Loading conversation data from {data_file}")
        
        # Load data based on file type
        if data_file.endswith('.json'):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    processed_conversations = self._process_conversation_item(item)
                    self.data.extend(processed_conversations)
        else:
            # Plain text format with conversation markers
            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                conversations = content.split('\n\n')
                for conv in conversations:
                    if conv.strip():
                        self.data.append(conv.strip())
        
        logger.info(f"Loaded {len(self.data)} conversation samples")
        
        # Add special tokens to tokenizer if using sentencepiece
        if self.tokenizer_type == 'sentencepiece':
            self._extend_tokenizer()
    
    def _extend_tokenizer(self):
        """Add special conversation tokens to sentencepiece tokenizer"""
        # Note: In practice, you'd retrain the tokenizer or use a different approach
        # For now, we'll handle these in text preprocessing
        pass
    
    def _process_conversation_item(self, item: Dict) -> List[str]:
        """Process a single conversation item into training samples"""
        conversations = []
        
        if 'conversations' in item:
            # Multi-turn conversation format
            conversation_text = ""
            for turn in item['conversations']:
                role = turn['from']
                content = turn['value']
                if role == 'human' or role == 'user':
                    conversation_text += f"{self.user_token}\n{content}\n"
                elif role == 'gpt' or role == 'assistant':
                    conversation_text += f"{self.assistant_token}\n{content}\n{self.end_token}\n"
                elif role == 'system':
                    conversation_text = f"{self.system_token}\n{content}\n" + conversation_text
            
            conversations.append(conversation_text.strip())
            
        elif 'instruction' in item:
            # Instruction-following format
            instruction = item['instruction']
            response = item['output']
            context = item.get('input', '')
            
            if context:
                conversation_text = f"{self.user_token}\n{instruction}\n{context}\n{self.assistant_token}\n{response}\n{self.end_token}"
            else:
                conversation_text = f"{self.user_token}\n{instruction}\n{self.assistant_token}\n{response}\n{self.end_token}"
            
            conversations.append(conversation_text)
        
        return conversations
    
    def _tokenize_conversation(self, text: str):
        """Tokenize conversation with role information"""
        if self.tokenizer_type == 'sentencepiece':
            tokens = self.tokenizer.encode(text, out_type=int)
        else:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Create role IDs (0: pad, 1: user, 2: assistant, 3: system)
        role_ids = []
        current_role = 0  # default to pad
        
        # Parse text to identify roles
        lines = text.split('\n')
        for line in lines:
            if line.strip() == self.user_token:
                current_role = 1
            elif line.strip() == self.assistant_token:
                current_role = 2
            elif line.strip() == self.system_token:
                current_role = 3
            elif line.strip() == self.end_token:
                current_role = 0
            
            # Estimate tokens for this line (rough approximation)
            line_tokens = len(line.split()) if line.strip() else 1
            role_ids.extend([current_role] * line_tokens)
        
        # Ensure role_ids matches token length
        if len(role_ids) > len(tokens):
            role_ids = role_ids[:len(tokens)]
        elif len(role_ids) < len(tokens):
            role_ids.extend([0] * (len(tokens) - len(role_ids)))
        
        return tokens, role_ids
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        conversation = self.data[idx]
        
        # Tokenize with role information
        tokens, role_ids = self._tokenize_conversation(conversation)
        
        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            role_ids = role_ids[:self.max_length]
        
        # Create input and target (shifted by 1)
        input_ids = tokens[:-1] if len(tokens) > 1 else tokens
        target_ids = tokens[1:] if len(tokens) > 1 else tokens
        input_role_ids = role_ids[:-1] if len(role_ids) > 1 else role_ids
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        pad_length = self.max_length - 1 - len(input_ids)
        if pad_length > 0:
            input_ids.extend([0] * pad_length)
            target_ids.extend([0] * pad_length)
            input_role_ids.extend([0] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        # Create loss mask (only compute loss on assistant responses)
        loss_mask = []
        for i, role in enumerate(input_role_ids):
            if role == 2:  # assistant role
                loss_mask.append(1)
            else:
                loss_mask.append(0)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'role_ids': torch.tensor(input_role_ids, dtype=torch.long),
            'loss_mask': torch.tensor(loss_mask, dtype=torch.float)
        }

class ImprovedTrainer:
    """Enhanced trainer with conversation-specific optimizations"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Update config with vocabulary size
        if hasattr(self.tokenizer, '__len__'):
            self.config.vocab_size = len(self.tokenizer)
        else:
            self.config.vocab_size = self.tokenizer.vocab_size()
        
        # Initialize model
        self.model = ImprovedSinhalaLLM(self.config).to(self.device)
        logger.info(f"Model initialized with {self.count_parameters():,} parameters")
        
        # Setup optimizers with different learning rates for different components
        self.setup_optimizers()
        
        # Enhanced loss function
        self.setup_loss_functions()
        
        # Mixed precision training
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.conversation_metrics = []
    
    def load_tokenizer(self):
        """Load the trained tokenizer"""
        if self.config.tokenizer_type == 'sentencepiece':
            tokenizer_path = '../models/tokenizer/sinhala_sp.model'
            self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
        else:
            tokenizer_path = '../models/tokenizer/sinhala_tokenizer'
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        logger.info(f"Loaded {self.config.tokenizer_type} tokenizer")
    
    def setup_optimizers(self):
        """Setup optimizers with different learning rates for different components"""
        # Separate parameters for different components
        embedding_params = list(self.model.embeddings.parameters())
        transformer_params = list(self.model.layers.parameters()) + list(self.model.final_layer_norm.parameters())
        head_params = list(self.model.lm_head.parameters())
        
        # Different learning rates for different components
        param_groups = [
            {'params': embedding_params, 'lr': self.config.embedding_lr, 'weight_decay': 0.01},
            {'params': transformer_params, 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': head_params, 'lr': self.config.head_lr, 'weight_decay': 0.01}
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.warmup_epochs,
            T_mult=2,
            eta_min=self.config.learning_rate * 0.01
        )
    
    def setup_loss_functions(self):
        """Setup enhanced loss functions for conversation training"""
        # Primary language modeling loss
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        
        # Conversation coherence loss (optional)
        self.coherence_weight = 0.1
    
    def compute_loss(self, logits, targets, loss_mask=None):
        """Compute enhanced loss with conversation awareness"""
        # Reshape for loss computation
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        
        # Compute token-level losses
        token_losses = self.lm_criterion(flat_logits, flat_targets)
        
        if loss_mask is not None:
            # Apply loss mask to focus on assistant responses
            flat_loss_mask = loss_mask.view(-1)
            masked_losses = token_losses * flat_loss_mask
            
            # Compute average loss only over non-masked tokens
            num_valid_tokens = flat_loss_mask.sum()
            if num_valid_tokens > 0:
                loss = masked_losses.sum() / num_valid_tokens
            else:
                loss = masked_losses.mean()
        else:
            loss = token_losses.mean()
        
        return loss
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def create_dataloaders(self):
        """Create training and validation dataloaders"""
        # Load datasets
        train_dataset = ConversationDataset(
            '../data/processed/train_conversations.txt',
            self.tokenizer,
            self.config.max_seq_length,
            self.config.tokenizer_type
        )
        
        val_dataset = ConversationDataset(
            '../data/processed/val_conversations.txt',
            self.tokenizer,
            self.config.max_seq_length,
            self.config.tokenizer_type
        )
        
        # Create dataloaders with improved settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=False
        )
        
        logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with improved techniques"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            role_ids = batch['role_ids'].to(self.device)
            loss_mask = batch['loss_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # Mixed precision training
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        role_ids=role_ids,
                        return_dict=True
                    )
                    logits = outputs['logits']
                    loss = self.compute_loss(logits, target_ids, loss_mask)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    role_ids=role_ids,
                    return_dict=True
                )
                logits = outputs['logits']
                loss = self.compute_loss(logits, target_ids, loss_mask)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # Log every 50 batches
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model with conversation-specific metrics"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                role_ids = batch['role_ids'].to(self.device)
                loss_mask = batch['loss_mask'].to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            role_ids=role_ids,
                            return_dict=True
                        )
                        logits = outputs['logits']
                        loss = self.compute_loss(logits, target_ids, loss_mask)
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        role_ids=role_ids,
                        return_dict=True
                    )
                    logits = outputs['logits']
                    loss = self.compute_loss(logits, target_ids, loss_mask)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def generate_conversation_sample(self, prompt, max_length=200):
        """Generate a conversation sample for evaluation"""
        self.model.eval()
        
        # Format prompt for conversation
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Tokenize prompt
        if self.config.tokenizer_type == 'sentencepiece':
            tokens = self.tokenizer.encode(formatted_prompt, out_type=int)
        else:
            tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate_chat_response(
                input_ids,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True
            )
        
        # Decode generated text
        if self.config.tokenizer_type == 'sentencepiece':
            generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        else:
            generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        
        return generated_text
    
    def save_checkpoint(self, epoch, val_loss, best_val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'config': self.config.__dict__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, '../models/improved_sinhala_llm_latest.pt')
        
        # Save best checkpoint
        if val_loss <= best_val_loss:
            torch.save(checkpoint, '../models/improved_sinhala_llm_best.pt')
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        logger.info(f"Checkpoint saved at epoch {epoch+1}")
    
    def train(self):
        """Main training loop with enhanced techniques"""
        logger.info("Starting improved training...")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        
        best_val_loss = float('inf')
        patience = self.config.patience
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, best_val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
            
            # Generate sample every 3 epochs
            if (epoch + 1) % 3 == 0:
                sample_prompt = "ශ්‍රී ලංකාවේ ප්‍රධාන නගර කුමක්ද?"
                generated = self.generate_conversation_sample(sample_prompt)
                logger.info(f"Sample generation:\n{generated[:500]}...")
        
        logger.info("Improved training completed!")

def main():
    """Main function with improved configuration"""
    # Enhanced configuration - adjusted for CPU training
    config = ImprovedModelConfig(
        vocab_size=32000,  # Will be updated after loading tokenizer
        hidden_size=512,   # Reduced from 1024 for faster CPU training
        num_attention_heads=8,  # Reduced from 16
        num_layers=12,     # Reduced from 24 for faster training
        intermediate_size=2048,  # Reduced from 4096
        max_position_embeddings=4096,
        dropout=0.1,
        attention_dropout=0.1,
        use_rope=True,
    )
    
    # Add training specific attributes
    config.batch_size = 2  # Reduced for CPU training
    config.learning_rate = 3e-4
    config.embedding_lr = 1e-4  # Lower LR for embeddings
    config.head_lr = 5e-4  # Higher LR for head
    config.weight_decay = 0.01
    config.max_grad_norm = 1.0
    config.num_epochs = 10  # Reduced for testing
    config.warmup_epochs = 2
    config.patience = 3
    config.tokenizer_type = 'sentencepiece'
    config.max_seq_length = 1024  # Reduced from 2048
    
    # Create trainer and start training
    trainer = ImprovedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 