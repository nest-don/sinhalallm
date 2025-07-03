import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for Sinhala LLM"""
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 3
    eos_token_id: int = 1

class SinhalaEmbeddings(nn.Module):
    """Embeddings for tokens and positions"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
        # Layer normalization and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # Register position_ids buffer
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
    
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings
        embeddings = word_embeddings + position_embeddings
        
        # Apply layer norm and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class SinhalaAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, config: ModelConfig):
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
        self.dropout = nn.Dropout(config.dropout)
    
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
        use_cache: bool = False
    ):
        # Linear transformations
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
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
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply causal mask (for autoregressive generation)
        seq_length = hidden_states.size(1)
        causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=hidden_states.device))
        causal_mask = causal_mask.view(1, 1, seq_length, seq_length)
        attention_scores = attention_scores.masked_fill(causal_mask == 0, -float('inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # Reshape
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # Output projection
        attention_output = self.dense(context_layer)
        
        return attention_output, present_key_value

class SinhalaFeedForward(nn.Module):
    """Feed-forward network"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_1(hidden_states)
        hidden_states = F.gelu(hidden_states)  # GELU activation
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states

class SinhalaLayer(nn.Module):
    """Single transformer layer"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = SinhalaAttention(config)
        self.feed_forward = SinhalaFeedForward(config)
        
        # Layer normalization
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ):
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        attention_output, present_key_value = self.attention(
            hidden_states, 
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = residual + self.dropout(attention_output)
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.output_norm(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(feed_forward_output)
        
        return hidden_states, present_key_value

class SinhalaEncoder(nn.Module):
    """Stack of transformer layers"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            SinhalaLayer(config) for _ in range(config.num_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False
    ):
        all_hidden_states = []
        present_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)
            
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        return hidden_states, present_key_values

class SinhalaLMHead(nn.Module):
    """Language modeling head"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits

class SinhalaLLM(nn.Module):
    """Complete Sinhala Language Model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Model components
        self.embeddings = SinhalaEmbeddings(config)
        self.encoder = SinhalaEncoder(config)
        self.lm_head = SinhalaLMHead(config)
        
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
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ):
        # Get embeddings
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        
        # Create attention mask if not provided
        if attention_mask is not None:
            # Convert attention mask to the format expected by the model
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Pass through encoder
        hidden_states, present_key_values = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': present_key_values,
            'hidden_states': hidden_states
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: int = None
    ):
        """Generate text using the model"""
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Generate tokens one by one
        past_key_values = None
        generated = input_ids
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(
                input_ids=generated[:, -1:] if past_key_values is not None else generated,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            logits = outputs['logits'][:, -1, :]  # Get last token logits
            past_key_values = outputs['past_key_values']
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample next token
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Stop if EOS token is generated
            if next_token.item() == self.config.eos_token_id:
                break
        
        return generated

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Example usage
if __name__ == "__main__":
    # Create model configuration
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048
    )
    
    # Create model
    model = SinhalaLLM(config)
    
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    batch_size = 2
    seq_length = 50
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    outputs = model(input_ids)
    print(f"Output logits shape: {outputs['logits'].shape}")
    
    # Test generation
    test_input = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(test_input, max_length=20)
    print(f"Generated sequence shape: {generated.shape}") 