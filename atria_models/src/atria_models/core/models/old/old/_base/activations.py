"""
Simple modular transformer architecture.

Philosophy:
- Encoder models (BERT): bidirectional attention, no causal mask
- Decoder models (GPT): causal attention
- Task-specific heads are separate from the base model
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
import math


@dataclass
class ModelConfig:
    """Minimal transformer config."""
    vocab_size: int = 30522
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_size: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12


# ============================================================================
# Basic Building Blocks
# ============================================================================

class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, config: ModelConfig, causal: bool = False):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.causal = causal
        
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        if causal:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                .view(1, 1, config.max_seq_len, config.max_seq_len)
            )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # qkv projection
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, num_heads, T, head_dim)
        
        # attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # apply causal mask if needed
        if self.causal:
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # apply padding mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # apply attention to values
        out = attn @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)
        
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.ffn_size)
        self.fc2 = nn.Linear(config.ffn_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    """Single transformer block: attention + ffn with residual connections."""
    
    def __init__(self, config: ModelConfig, causal: bool = False):
        super().__init__()
        self.attn = Attention(config, causal=causal)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # pre-norm architecture
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


# ============================================================================
# Base Models
# ============================================================================

class EncoderModel(nn.Module):
    """Base encoder model (BERT-style): bidirectional attention."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            Block(config, causal=False) for _ in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = input_ids.shape
        
        # embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)
        
        # prepare mask for attention (B, 1, 1, T)
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]
        else:
            mask = None
        
        # apply blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        return x


class DecoderModel(nn.Module):
    """Base decoder model (GPT-style): causal attention."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # transformer blocks with causal masking
        self.blocks = nn.ModuleList([
            Block(config, causal=True) for _ in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = input_ids.shape
        
        # embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)
        
        # prepare mask for attention (B, 1, 1, T)
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]
        else:
            mask = None
        
        # apply blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        return x


# ============================================================================
# Concrete Models with Task-Specific Heads
# ============================================================================

class BERT(nn.Module):
    """BERT: encoder with optional pooling for classification."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = EncoderModel(config)
        
        # optional: pooler for classification tasks
        self.pooler = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_pooled: bool = False
    ):
        hidden_states = self.encoder(input_ids, attention_mask)
        
        if return_pooled:
            # pool first token for classification
            pooled = self.pooler(hidden_states[:, 0])
            return hidden_states, pooled
        
        return hidden_states


class GPT(nn.Module):
    """GPT: decoder with language modeling head."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.decoder = DecoderModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # tie weights
        self.lm_head.weight = self.decoder.token_embed.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        hidden_states = self.decoder(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss


class BERTForMaskedLM(nn.Module):
    """BERT with MLM head - separate from base BERT."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.bert = BERT(config)
        
        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size)
        )
        
        # tie weights
        self.mlm_head[-1].weight = self.bert.encoder.token_embed.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        hidden_states = self.bert(input_ids, attention_mask)
        logits = self.mlm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss


class BERTForSequenceClassification(nn.Module):
    """BERT for classification tasks."""
    
    def __init__(self, config: ModelConfig, num_labels: int = 2):
        super().__init__()
        self.bert = BERT(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        _, pooled = self.bert(input_ids, attention_mask, return_pooled=True)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        
        return logits, loss


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    config = ModelConfig(
        vocab_size=30522,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
    )
    
    # BERT example
    print("=== BERT ===")
    bert = BERT(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 128))
    hidden = bert(input_ids)
    print(f"Hidden states: {hidden.shape}")
    
    # GPT example
    print("\n=== GPT ===")
    gpt = GPT(config)
    logits, _ = gpt(input_ids)
    print(f"Logits: {logits.shape}")
    
    # BERT for MLM
    print("\n=== BERT for MLM ===")
    bert_mlm = BERTForMaskedLM(config)
    mlm_logits, _ = bert_mlm(input_ids)
    print(f"MLM logits: {mlm_logits.shape}")
    
    # Parameter counts
    print("\n=== Parameter Counts ===")
    print(f"BERT: {sum(p.numel() for p in bert.parameters()):,}")
    print(f"GPT: {sum(p.numel() for p in gpt.parameters()):,}")