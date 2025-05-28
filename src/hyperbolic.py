import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class LorentzManifold:
    """Lorentz model of hyperbolic space operations"""
    
    def __init__(self, dim: int, curvature: float = -1.0):
        self.dim = dim
        self.curvature = curvature
        self.scale = 1. / np.sqrt(-curvature)
        
    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at point x for tangent vector v"""
        v_norm = torch.clamp(self.inner(x, v, keepdim=True), min=1e-6)
        v_norm = torch.sqrt(v_norm)
        return torch.cosh(v_norm) * x + torch.sinh(v_norm) * (v / v_norm)
    
    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """Project to hyperboloid manifold"""
        return x / torch.sqrt(-self.inner(x, x, keepdim=True))
    
    def inner(self, x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Lorentz inner product"""
        m = x * y
        if keepdim:
            return (m[..., 1:] - m[..., :1]).sum(dim=-1, keepdim=True)
        return (m[..., 1:] - m[..., :1]).sum(dim=-1)
    
    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance between points"""
        return self.scale * torch.acosh(torch.clamp(-self.inner(x, y), min=1+1e-6))
    
    def parallel_transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Parallel transport of v from x to y"""
        alpha = -self.inner(x, y)
        return v + self.inner(y, v) / (1 - alpha) * (x + y)


class HyperbolicCausalEncoder(nn.Module):
    """Encodes causal graph structure into hyperbolic space"""
    
    def __init__(self, manifold_dim: int, curvature: float = -1.0):
        super().__init__()
        self.manifold = LorentzManifold(manifold_dim, curvature)
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, adj_matrix: torch.Tensor, node_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            adj_matrix: [num_nodes, num_nodes] adjacency matrix
            node_features: [num_nodes, feature_dim] optional features
            
        Returns:
            [num_nodes, manifold_dim] hyperbolic embeddings
        """
        num_nodes = adj_matrix.size(0)
        
        # Initialize embeddings in tangent space
        if node_features is not None:
            features = F.normalize(node_features, p=2, dim=-1)
            tangent_emb = torch.cat([torch.zeros(num_nodes, 1), features], dim=-1)
        else:
            tangent_emb = torch.cat([torch.zeros(num_nodes, 1), 
                                   torch.randn(num_nodes, self.manifold.dim-1)], dim=-1)
        
        # Project to manifold
        hyp_emb = self.manifold.proj(tangent_emb)
        
        # Add hierarchy through root connection
        root = torch.zeros(1, self.manifold.dim)
        root[0, 0] = 1.0
        hyp_emb = torch.cat([root, hyp_emb], dim=0)
        
        return hyp_emb


class RSGD(torch.optim.Optimizer):
    """Riemannian Stochastic Gradient Descent"""
    
    def __init__(self, params, manifold: LorentzManifold, lr: float = 1e-3):
        defaults = dict(lr=lr, manifold=manifold)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            manifold = group['manifold']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Riemannian gradient
                grad = p.grad.data
                if grad.is_sparse:
                    grad = grad.to_dense()
                
                # Project to tangent space
                tangent_grad = grad - (manifold.inner(p.data, grad) / 
                                     manifold.inner(p.data, p.data)) * p.data
                
                # Exponential map update
                p.data.copy_(manifold.expmap(p.data, -lr * tangent_grad))
                
        return loss


class HyperbolicCausalModel(nn.Module):
    """End-to-end hyperbolic causal structure learning"""
    
    def __init__(self, num_nodes: int, feature_dim: int, manifold_dim: int = 32):
        super().__init__()
        self.encoder = HyperbolicCausalEncoder(manifold_dim)
        self.manifold = LorentzManifold(manifold_dim)
        
        # Structural parameters
        self.adj_logits = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.register_buffer('mask', torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get hyperbolic embeddings
        hyp_emb = self.encoder(None, x)  # [num_nodes+1, manifold_dim]
        
        # Compute hyperbolic distances
        root = hyp_emb[:1]  # root node
        node_emb = hyp_emb[1:]
        
        # Pairwise distances
        dists = self.manifold.dist(node_emb.unsqueeze(1), node_emb.unsqueeze(0))
        
        # Combine with learned adjacency
        adj_logits = self.adj_logits * self.mask
        adj_probs = torch.sigmoid(adj_logits - dists.detach())
        
        return adj_probs, hyp_emb




# Usage Example
if __name__ == "__main__":
    num_nodes = 10
    feature_dim = 16
    manifold_dim = 8
    
    model = HyperbolicCausalModel(num_nodes, feature_dim, manifold_dim)
    optimizer = RSGD(model.parameters(), model.manifold, lr=0.01)
    
    # Sample data
    x = torch.randn(num_nodes, feature_dim)
    adj_matrix = torch.bernoulli(0.3 * torch.ones(num_nodes, num_nodes))
    
    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        adj_pred, embeddings = model(x)
        
        # Reconstruction loss
        loss = F.binary_cross_entropy(adj_pred, adj_matrix)
        
        # Add hyperbolic regularization
        root = embeddings[:1]
        nodes = embeddings[1:]
        hyp_reg = model.manifold.dist(root, nodes).mean()
        
        total_loss = loss + 0.1 * hyp_reg
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")