#!/usr/bin/env python3
"""
Improved Sinhala LLM Architecture
Enhanced for better chat performance and coherence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class ImprovedModelConfig:
    """Improved configuration for Sinhala LLM with better chat capabilities"""
    vocab_size: int = 32000
    hidden_size: int = 1024  # Increased from 768
    num_layers: int = 24     # Increased from 12
    num_attention_heads: int = 16  # Increased from 12
    intermediate_size: int = 4096   # Increased from 3072
    max_position_embeddings: int = 4096  # Increased from 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 3
    eos_token_id: int = 1
    user_token_id: int = 32001  # Special token for user
    assistant_token_id: int = 32002  # Special token for assistant
    end_turn_token_id: int = 32003  # Special token for end of turn
    # Chat-specific configurations
    conversation_max_turns: int = 10
    response_max_length: int = 512
    use_rope: bool = True  # Rotary Position Embedding
    use_flash_attention: bool = True

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding for better long-range attention"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
    def forward(self, seq_len: int, device: torch.device):
        # Generate position indices
        position = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        # Compute the frequencies
        freqs = torch.outer(position, self.inv_freq)
        
        # Create sin and cos embeddings
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        return cos_emb, sin_emb

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys"""
    # q and k shape: [batch_size, num_heads, seq_len, head_dim]
    # cos and sin shape: [seq_len, head_dim]
    
    # Expand cos and sin to match q and k dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    
    # Split the head dimension in half for rotation
    d = q.shape[-1] // 2
    q_rot = q[..., :d]
    q_pass = q[..., d:]
    k_rot = k[..., :d]
    k_pass = k[..., d:]
    
    # Apply rotation: rotate_half is the key operation
    # For complex rotation: q * cos + rotate_half(q) * sin
    q_rot_rotated = q_rot * cos[..., :d] - torch.cat([q_rot[..., d//2:], q_rot[..., :d//2]], dim=-1) * sin[..., :d]
    k_rot_rotated = k_rot * cos[..., :d] - torch.cat([k_rot[..., d//2:], k_rot[..., :d//2]], dim=-1) * sin[..., :d]
    
    # Concatenate rotated and non-rotated parts
    q = torch.cat([q_rot_rotated, q_pass], dim=-1)
    k = torch.cat([k_rot_rotated, k_pass], dim=-1)
    
    return q, k

class ImprovedAttention(nn.Module):
    """Enhanced multi-head attention with RoPE and conversation awareness"""
    
    def __init__(self, config: ImprovedModelConfig):
        super().__init__()
        self.config = config
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by "
                f"number of attention heads {config.num_attention_heads}"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear layers for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output projection
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # RoPE
        if config.use_rope:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.attention_head_size, config.max_position_embeddings
            )
        
        # Scaling factor
        self.scale = math.sqrt(self.attention_head_size)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for attention computation"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None
    ):
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Linear transformations
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Apply RoPE if enabled
        if self.config.use_rope:
            cos, sin = self.rotary_emb(seq_len, hidden_states.device)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)
        
        # Handle past key values for caching
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_layer = torch.cat([past_key, key_layer], dim=-2)
            value_layer = torch.cat([past_value, value_layer], dim=-2)
        
        # Store key-value for next iteration if using cache
        if use_cache:
            present_key_value = (key_layer, value_layer)
        else:
            present_key_value = None
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / self.scale
        
        # Apply causal mask (for autoregressive generation)
        current_seq_len = attention_scores.size(-1)
        causal_mask = torch.tril(torch.ones(current_seq_len, current_seq_len, device=hidden_states.device))
        causal_mask = causal_mask.view(1, 1, current_seq_len, current_seq_len)
        attention_scores = attention_scores.masked_fill(causal_mask == 0, -float('inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Reshape
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # Output projection
        attention_output = self.dense(context_layer)
        attention_output = self.output_dropout(attention_output)
        
        return attention_output, present_key_value

class ImprovedFeedForward(nn.Module):
    """Enhanced feed-forward network with SwiGLU activation"""
    
    def __init__(self, config: ImprovedModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: gate * silu(up) 
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states = gate * F.silu(up)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.down_proj(hidden_states)
        return hidden_states

class ImprovedTransformerLayer(nn.Module):
    """Enhanced transformer layer with pre-layer normalization"""
    
    def __init__(self, config: ImprovedModelConfig):
        super().__init__()
        self.attention = ImprovedAttention(config)
        self.feed_forward = ImprovedFeedForward(config)
        
        # Pre-layer normalization (more stable training)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None
    ):
        # Pre-attention layer norm
        normed_hidden_states = self.input_layernorm(hidden_states)
        
        # Self-attention with residual connection
        attention_output, present_key_value = self.attention(
            normed_hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_ids=position_ids
        )
        hidden_states = hidden_states + attention_output
        
        # Pre-FFN layer norm
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Feed-forward with residual connection
        ffn_output = self.feed_forward(normed_hidden_states)
        hidden_states = hidden_states + ffn_output
        
        return hidden_states, present_key_value

class ConversationAwareEmbeddings(nn.Module):
    """Enhanced embeddings with conversation-aware features"""
    
    def __init__(self, config: ImprovedModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Position embeddings (only if not using RoPE)
        if not config.use_rope:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings,
                config.hidden_size
            )
        
        # Role embeddings for chat (user/assistant/system)
        self.role_embeddings = nn.Embedding(4, config.hidden_size)  # 4 roles: pad, user, assistant, system
        
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        if not config.use_rope:
            # Register position_ids buffer
            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1))
            )
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None
    ):
        seq_length = input_ids.size(1)
        
        # Get word embeddings
        word_embeddings = self.word_embeddings(input_ids)
        
        # Add position embeddings if not using RoPE
        if not self.config.use_rope:
            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings
        
        # Add role embeddings if provided
        if role_ids is not None:
            role_embeddings = self.role_embeddings(role_ids)
            embeddings = embeddings + role_embeddings
        
        # Apply layer norm and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class ImprovedSinhalaLLM(nn.Module):
    """Improved Sinhala LLM with enhanced chat capabilities"""
    
    def __init__(self, config: ImprovedModelConfig):
        super().__init__()
        self.config = config
        
        # Model components
        self.embeddings = ConversationAwareEmbeddings(config)
        self.layers = nn.ModuleList([
            ImprovedTransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        role_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True
    ):
        # Get embeddings
        hidden_states = self.embeddings(
            input_ids, 
            position_ids=position_ids,
            role_ids=role_ids
        )
        
        # Create attention mask if not provided
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Pass through transformer layers
        present_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                position_ids=position_ids
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'past_key_values': present_key_values,
                'hidden_states': hidden_states
            }
        else:
            return (loss, logits, present_key_values, hidden_states)
    
    def generate_chat_response(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ):
        """Enhanced generation method optimized for chat conversations"""
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.end_turn_token_id
        
        batch_size, input_length = input_ids.shape
        device = input_ids.device
        
        # Initialize generation variables
        past_key_values = None
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Token frequency tracker for repetition penalty
        token_counts = torch.zeros(batch_size, self.config.vocab_size, device=device)
        
        for step in range(max_new_tokens):
            # Prepare inputs for current step
            if past_key_values is not None:
                # Only use the last token for input
                step_input_ids = generated[:, -1:]
            else:
                step_input_ids = generated
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=step_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
            
            next_token_logits = outputs['logits'][:, -1, :]
            past_key_values = outputs['past_key_values']
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for batch_idx in range(batch_size):
                    for token_id in generated[batch_idx]:
                        next_token_logits[batch_idx, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < top_k_logits[:, [-1]]] = -float('inf')
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Update token counts
            for batch_idx in range(batch_size):
                token_counts[batch_idx, next_tokens[batch_idx]] += 1
            
            # Update finished status
            finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=-1)
            
            # Update attention mask if provided
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                ], dim=-1)
            
            # Stop if all sequences are finished
            if finished.all():
                break
        
        return generated

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage
if __name__ == "__main__":
    # Create improved model configuration
    config = ImprovedModelConfig(
        vocab_size=32000,
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=4096
    )
    
    # Create model
    model = ImprovedSinhalaLLM(config)
    
    print(f"Improved model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_length = 100
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    outputs = model(input_ids)
    print(f"Output logits shape: {outputs['logits'].shape}") 