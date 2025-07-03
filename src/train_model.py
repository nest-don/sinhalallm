#!/usr/bin/env python3
"""
Sinhala LLM Training Script
Train a custom transformer model on prepared Sinhala data
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

# Import our custom model
from model_architecture import SinhalaLLM, ModelConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SinhalaDataset(Dataset):
    """Dataset class for Sinhala text data"""
    
    def __init__(self, data_file, tokenizer, max_length=512, tokenizer_type='sentencepiece'):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer_type = tokenizer_type
        
        logger.info(f"Loading data from {data_file}")
        
        # Load data based on file type
        if data_file.endswith('.json'):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    if 'conversations' in item:
                        # Conversation format
                        conversation_text = ""
                        for turn in item['conversations']:
                            role = turn['from']
                            content = turn['value']
                            if role == 'human':
                                conversation_text += f"<|user|>{content}"
                            else:
                                conversation_text += f"<|assistant|>{content}<|end|>"
                        self.data.append(conversation_text)
                    elif 'instruction' in item:
                        # Alpaca format
                        instruction = item['instruction']
                        response = item['output']
                        text = f"<|user|>{instruction}<|assistant|>{response}<|end|>"
                        self.data.append(text)
        else:
            # Plain text format
            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Split into chunks
                chunks = content.split('\n\n')
                self.data.extend([chunk.strip() for chunk in chunks if chunk.strip()])
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize based on tokenizer type
        if self.tokenizer_type == 'sentencepiece':
            tokens = self.tokenizer.encode(text, out_type=int)
        else:  # huggingface
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input and target (shifted by 1 for language modeling)
        input_ids = tokens[:-1] if len(tokens) > 1 else tokens
        target_ids = tokens[1:] if len(tokens) > 1 else tokens
        
        # Pad to max_length - 1 (since we shift)
        pad_length = self.max_length - 1 - len(input_ids)
        if pad_length > 0:
            input_ids.extend([0] * pad_length)  # 0 is pad token
            target_ids.extend([0] * pad_length)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1] * len(input_ids), dtype=torch.float)
        }

class SinhalaTrainer:
    """Trainer class for Sinhala LLM"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Update config with vocabulary size
        self.config.vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else self.tokenizer.vocab_size()
        
        # Initialize model
        self.model = SinhalaLLM(self.config).to(self.device)
        logger.info(f"Model initialized with {self.count_parameters()} parameters")
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        
        # Mixed precision training
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def load_tokenizer(self):
        """Load the trained tokenizer"""
        if self.config.tokenizer_type == 'sentencepiece':
            self.tokenizer = spm.SentencePieceProcessor(model_file='sinhala_sp.model')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('sinhala_tokenizer')
        
        logger.info(f"Loaded {self.config.tokenizer_type} tokenizer")
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def create_dataloaders(self):
        """Create training and validation dataloaders"""
        # Load datasets
        train_dataset = SinhalaDataset(
            'train_conversations.txt', 
            self.tokenizer, 
            self.config.max_seq_length,
            self.config.tokenizer_type
        )
        
        val_dataset = SinhalaDataset(
            'val_conversations.txt',
            self.tokenizer,
            self.config.max_seq_length,
            self.config.tokenizer_type
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                # Mixed precision training
                with autocast():
                    outputs = self.model(input_ids)
                    logits = outputs['logits']
                    loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training
                outputs = self.model(input_ids)
                logits = outputs['logits']
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = self.model(input_ids)
                        logits = outputs['logits']
                        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                else:
                    outputs = self.model(input_ids)
                    logits = outputs['logits']
                    loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
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
        torch.save(checkpoint, 'sinhala_llm_latest.pt')
        
        # Save best checkpoint
        if val_loss <= best_val_loss:
            torch.save(checkpoint, 'sinhala_llm_best.pt')
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        logger.info(f"Checkpoint saved at epoch {epoch+1}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate curve
        ax2.plot(epochs, self.learning_rates, 'g-')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Training curves saved as training_curves.png")
    
    def generate_sample(self, prompt, max_length=100):
        """Generate a sample text from the model"""
        self.model.eval()
        
        # Tokenize prompt
        if self.config.tokenizer_type == 'sentencepiece':
            tokens = self.tokenizer.encode(prompt, out_type=int)
        else:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids, 
                max_length=max_length,
                temperature=0.8,
                do_sample=True
            )
        
        # Decode generated text
        if self.config.tokenizer_type == 'sentencepiece':
            generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        else:
            generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
        
        return generated_text
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()
        
        best_val_loss = float('inf')
        patience = 3
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
            
            # Generate sample every 5 epochs
            if (epoch + 1) % 5 == 0:
                sample_prompt = "ශ්‍රී ලංකාව"
                generated = self.generate_sample(sample_prompt)
                logger.info(f"Sample generation: {generated[:200]}...")
        
        # Plot training curves
        self.plot_training_curves()
        
        logger.info("Training completed!")

def main():
    """Main function"""
    # Configuration
    config = ModelConfig(
        vocab_size=32000,  # Will be updated after loading tokenizer
        hidden_size=512,
        num_attention_heads=8,
        num_layers=6,
        intermediate_size=2048,
        max_position_embeddings=512,
        dropout=0.1,
    )
    
    # Add training specific attributes to config
    config.batch_size = 4  # Small batch size for limited resources
    config.learning_rate = 5e-4
    config.weight_decay = 0.01
    config.num_epochs = 20
    config.tokenizer_type = 'sentencepiece'  # or 'huggingface'
    config.max_seq_length = 512
    
    # Create trainer and start training
    trainer = SinhalaTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 