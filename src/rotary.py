import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class CAPE(nn.Module):
    """
    Causal-Aware Positional Encoding (CAPE) implementation.
    
    This module applies CAPE to input tensors.
    
    Args:
        dim: Dimension of the embeddings
        c: Scalar parameter
        max_seq_len: Maximum sequence length        
    """

    def __init__(
        self,
        dim: int,
        c: float = math.pi/4,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.c = c
        self.max_seq_len = max_seq_len

    def _compute_rotary(
        self, 
        e: torch.Tensor,
        head_dim: int,
        seq_len: int = 2048,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute the rotary form of positional encoding."""
        _, e_seq_len, e_head_dim = e.shape

        # 1. Check input shape
        if e.dim() != 3:
            raise ValueError(
                f"Input positional encoding tensor must be 3D (batch, seq, head_dim//2), got {e.dim()}D"
            )

        batch_size, e_seq_len, e_head_dim = e.shape
        
        # 2. Check head dimension
        if e_head_dim != head_dim//2:
            raise ValueError(
                f"Input e_head_dim ({e_head_dim}) does not match expected head_dim ({head_dim//2})"
            )

        # 3. Check seq_len
        if e_seq_len+offset != seq_len:
            raise ValueError(
                f"Input e_seq_len ({e_seq_len}) does not match expected seq_len ({seq_len})"
            )
        
        e = e * self.c
        cos = e.cos().view(batch_size, e_seq_len, 1, e_head_dim)
        sin = e.sin().view(batch_size, e_seq_len, 1, e_head_dim)
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # (batch_size, e_seq_len, 1, 2*e_head_dim)
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # (batch_size, e_seq_len, 1, 2*e_head_dim)

        cos_padded = F.pad(cos, (0, 0, 0, 0, offset, 0))  # (batch_size, offset + e_seq_len, 1, head_dim)
        sin_padded = F.pad(sin, (0, 0, 0, 0, offset, 0))  # (batch_size, offset + e_seq_len, 1, head_dim)
        return cos_padded, sin_padded

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        offset: int = 0
    ) -> torch.Tensor:
        """
        Apply rotary position embeddings to input tensor.
        
        Args:
            x: Input data tensor of shape (batch_size, seq_len, n_heads, head_dim)
            e: Positional encoding tensor of shape (batch_size, seq_len, head_dim//2)
            offset: Position offset for the sequence

        Returns:
            Tensor with rotary embeddings applied
        """
        _, seq_len, _, head_dim = x.shape

        cos, sin = self._compute_rotary(e, head_dim, seq_len, offset)
        
        # Reshape input to separate real and imaginary parts
        x_rot = x.view(*x.shape[:-1], -1, 2)
        x_rot = torch.stack((-x_rot[..., 1], x_rot[..., 0]), dim=-1)
        x_rot = x_rot.view(*x.shape)
        print(x_rot.shape)
        
        # Apply rotary embeddings
        x_embed = (x * cos) + (x_rot * sin)
        
        return x_embed.type_as(x)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        e_q: torch.Tensor,
        e_k: torch.Tensor,
        offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to queries and keys.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, n_heads, head_dim)
            k: Key tensor of shape (batch_size, seq_len, n_heads, head_dim)
            e_q: Positional encoding tensor for q of shape (batch_size, seq_len, head_dim//2)
            e_k: Positional encoding tensor for k of shape (batch_size, seq_len, head_dim//2)
            offset: Position offset for the sequence
            
        Returns:
            Tuple of (q_rotated, k_rotated) with rotary embeddings applied
        """
        q_rotated = self.apply_rotary_emb(q, e_q, offset)
        k_rotated = self.apply_rotary_emb(k, e_k, offset)
        return q_rotated, k_rotated




if __name__ == "__main__":
    # Test configuration
    batch_size = 2
    seq_len = 10
    n_heads = 4
    head_dim = 32
    dim = n_heads * head_dim  # Total dimension
    offset = 3  # Position offset
    
    # Initialize CAPE
    cape = CAPE(dim=dim, c=math.pi/4, max_seq_len=2048)
    
    # Create test tensors
    q = torch.randn(batch_size, seq_len+offset, n_heads, head_dim).requires_grad_(True)
    k = torch.randn(batch_size, seq_len+offset, n_heads, head_dim).requires_grad_(True)
    e_q = torch.randn(batch_size, seq_len, head_dim//2)  # PE for queries
    e_k = torch.randn(batch_size, seq_len, head_dim//2)  # PE for keys
    
    print("=== Input Shapes ===")
    print(f"q shape: {q.shape}")
    print(f"k shape: {k.shape}")
    print(f"e_q shape: {e_q.shape}")
    print(f"e_k shape: {e_k.shape}")
    
    # Test forward pass
    q_rotated, k_rotated = cape.forward(q, k, e_q, e_k, offset=offset)
    
    print("\n=== Output Shapes ===")
    print(f"Rotated q shape: {q_rotated.shape}")
    print(f"Rotated k shape: {k_rotated.shape}")
    
    # Basic validation
    assert q_rotated.shape == q.shape, "Output q shape mismatch!"
    assert k_rotated.shape == k.shape, "Output k shape mismatch!"
    
    # Test gradient flow
    q_rotated.sum().backward()
    print("\nGradient test passed - backward() executed successfully")
    
    # Test with different sequence length
    try:
        bad_e = torch.randn(batch_size, seq_len+1, head_dim//2)
        cape(q, k, bad_e, e_k)  # Should raise ValueError
    except ValueError as e:
        print(f"\nSequence length check passed: {e}")
    
    print("\nAll tests passed!")