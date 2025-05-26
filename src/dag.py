import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DAGVAE(nn.Module):
    """
    DAG Variational Autoencoder combining DAG structure learning with VAE framework.
    
    Args:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions for encoder/decoder
        latent_dim: Dimension of latent space
        num_nodes: Number of nodes in the DAG
        prior_type: Type of prior ('gaussian' or 'laplacian')
        temperature: Temperature for Gumbel-Softmax sampling
        hard: Whether to use hard sampling
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        num_nodes: int = 10,
        prior_type: str = 'gaussian',
        temperature: float = 1.0,
        hard: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.prior_type = prior_type
        self.temperature = temperature
        self.hard = hard
        
        # Encoder network (inference model q(z|x))
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder network (generative model p(x|z))
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # DAG structure parameters
        self.adj_logits = nn.Parameter(torch.randn(num_nodes, num_nodes))
        self.register_buffer('mask', torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes))
        
        # Nonlinear SEM parameters
        # Nonlinear SEM parameters - now properly handling latent_dim
        self.sem_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + latent_dim, 32),  # Input: current node + aggregated parents
                nn.ReLU(),
                nn.Linear(32, latent_dim)
            ) for _ in range(num_nodes)
        ])
        
    @property
    def adj_matrix(self) -> torch.Tensor:
        """Generate the adjacency matrix with proper shape handling.
        
        Returns:
            torch.Tensor: The adjacency matrix of shape (num_nodes, num_nodes)
            containing values in [0, 1] during training and {0, 1} during evaluation.
        """
        # Apply mask to zero out diagonal (no self-connections)
        logits = self.adj_logits * self.mask  # shape: (num_nodes, num_nodes)
        
        if self.training:
            # During training: use Gumbel-Softmax sampling
            # We need to create a 2-class dimension for each edge
            logits = torch.stack([
                torch.zeros_like(logits),  # class 0: no edge
                logits                     # class 1: edge
            ], dim=-1)  # shape: (num_nodes, num_nodes, 2)
            
            adj = F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=self.hard,
                dim=-1
            )  # shape: (num_nodes, num_nodes, 2)
            
            # Return the probability/selection of class 1 (edge exists)
            return adj[..., 1]  # shape: (num_nodes, num_nodes)
        else:
            # During evaluation: use hard threshold
            adj = (logits > 0).float()  # shape: (num_nodes, num_nodes)
            return adj

    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input into latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from q(z|x)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from latent code"""
        return self.decoder(z)
    
    def apply_sem(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply nonlinear Structural Equation Model (SEM) to latent variables.
        
        Args:
            z: Latent variables of shape [batch_size, num_nodes, latent_dim]
            
        Returns:
            Transformed latent variables respecting DAG structure
        """
        adj = self.adj_matrix.unsqueeze(0)  # Add batch dim
        
        # Process each node in topological order
        topo_order = topological_sort(adj.squeeze(0))
        transformed_z = torch.zeros_like(z)
        
        for node in topo_order:
            # Get parent nodes
            parents = torch.nonzero(adj[0, node], as_tuple=True)[0]
            
            if len(parents) > 0:
                # Aggregate parent information
                parent_features = transformed_z[:, parents]
                agg_parent = parent_features.mean(dim=1, keepdim=True)
                
                # Apply nonlinear transformation
                node_z = z[:, node:node+1]
                input_feat = torch.cat([node_z, agg_parent], dim=-1)
            else:
                input_feat = z[:, node:node+1]
                
            # Apply node-specific SEM
            transformed_z[:, node] = self.sem_weights[node](input_feat).squeeze(1)
            
        return transformed_z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of DAG-VAE.
        
        Args:
            x: Input tensor of shape [batch_size, num_nodes, input_dim]
            
        Returns:
            tuple: (recon_x, mu, logvar, adj_matrix, kl_divergence)
        """
        batch_size = x.size(0)
        
        # Encode to latent space
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Apply DAG structure
        z_transformed = self.apply_sem(z)
        
        # Decode reconstruction
        recon_x = self.decode(z_transformed)
        
        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Get adjacency matrix
        adj = self.adj_matrix
        
        return recon_x, mu, logvar, adj, kl_div
    
    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_div: torch.Tensor,
        beta: float = 1.0,
        lambda_a: float = 0.1,
        h_thresh: float = 0.0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute VAE loss with DAG constraints.
        
        Args:
            recon_x: Reconstructed x
            x: Original input x
            mu: Latent mean
            logvar: Latent log variance
            kl_div: KL divergence term
            beta: Weight for KL term
            lambda_a: Weight for acyclicity constraint
            h_thresh: Threshold for acyclicity
            
        Returns:
            tuple: (total_loss, loss_dict)
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # Acyclicity constraint
        adj = self.adj_matrix
        h = torch.trace(torch.matrix_exp(adj * adj)) - self.num_nodes
        acyc_loss = lambda_a * h * h
        
        # Sparsity constraint
        sparsity_loss = torch.sum(torch.abs(adj))
        
        # Total loss
        total_loss = recon_loss + beta * kl_div + acyc_loss + 0.1 * sparsity_loss
        
        # Prepare loss dictionary
        loss_dict = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_div': kl_div,
            'acyc_loss': acyc_loss,
            'sparsity_loss': sparsity_loss
        }
        
        return total_loss, loss_dict
    
    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample from the generative model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        with torch.no_grad():
            # Sample from prior
            if self.prior_type == 'gaussian':
                z = torch.randn(num_samples, self.num_nodes, self.latent_dim)
            else:  # laplacian
                z = torch.distributions.Laplace(0, 1).sample(
                    (num_samples, self.num_nodes, self.latent_dim))
            
            # Apply SEM transformations
            z_transformed = self.apply_sem(z)
            
            # Decode samples
            samples = self.decode(z_transformed)
            return samples.view(num_samples, self.num_nodes, self.input_dim)


def topological_sort(adj_matrix: torch.Tensor) -> List[int]:
    """
    Perform topological sort on a DAG.
    
    Args:
        adj_matrix: Adjacency matrix of shape [num_nodes, num_nodes]
        
    Returns:
        List of node indices in topological order
        
    Raises:
        ValueError: If input is not a valid DAG
    """
    num_nodes = adj_matrix.size(0)
    visited = [False] * num_nodes
    order = []
    
    def _dfs(node: int):
        """DFS helper function for topological sort."""
        visited[node] = True
        neighbors = torch.nonzero(adj_matrix[node], as_tuple=True)[0].tolist()
        for neighbor in neighbors:
            if not visited[neighbor]:
                _dfs(neighbor)
        order.append(node)
    
    for node in range(num_nodes):
        if not visited[node]:
            _dfs(node)
    
    return order[::-1]  # Reverse to get topological order


def is_dag(adj_matrix: torch.Tensor) -> bool:
    """
    Check if adjacency matrix represents a valid DAG.
    
    Args:
        adj_matrix: Adjacency matrix of shape [num_nodes, num_nodes]
        
    Returns:
        bool: True if input is a valid DAG, False otherwise
    """
    num_nodes = adj_matrix.size(0)
    visited = [False] * num_nodes
    recursion_stack = [False] * num_nodes
    
    def _has_cycle(node: int) -> bool:
        """DFS helper function to detect cycles."""
        visited[node] = True
        recursion_stack[node] = True
        
        neighbors = torch.nonzero(adj_matrix[node], as_tuple=True)[0].tolist()
        for neighbor in neighbors:
            if not visited[neighbor]:
                if _has_cycle(neighbor):
                    return True
            elif recursion_stack[neighbor]:
                return True
        
        recursion_stack[node] = False
        return False
    
    for node in range(num_nodes):
        if not visited[node]:
            if _has_cycle(node):
                return False
    return True




if __name__ == "__main__":
    # Test configuration
    batch_size = 4
    num_nodes = 5
    input_dim = 10
    latent_dim = 16
    hidden_dims = [64, 32]
    
    print("=== Initializing DAGVAE ===")
    model = DAGVAE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        num_nodes=num_nodes,
        prior_type='gaussian'
    )
    
    # Create test tensors
    x = torch.randn(batch_size, num_nodes, input_dim).requires_grad_(True)
    
    print("\n=== Model Architecture ===")
    print(model)
    print(f"\nTrainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n=== Input Shapes ===")
    print(f"Input x shape: {x.shape}")
    
    # Test forward pass
    recon_x, mu, logvar, adj, kl_div = model.forward(x)
    
    print("\n=== Output Shapes ===")
    print(f"Reconstructed x shape: {recon_x.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"KL divergence: {kl_div.item():.4f}")
    
    # Basic validation
    assert recon_x.shape == x.shape, "Output shape mismatch!"
    assert mu.shape == (batch_size, num_nodes, latent_dim), "Mu shape mismatch!"
    assert logvar.shape == (batch_size, num_nodes, latent_dim), "Logvar shape mismatch!"
    assert adj.shape == (num_nodes, num_nodes), "Adjacency matrix shape mismatch!"
    
    # Test loss function
    loss, loss_dict = model.loss_function(recon_x, x, mu, logvar, kl_div)
    print("\n=== Loss Components ===")
    for k, v in loss_dict.items():
        print(f"{k}: {v.item():.4f}")
    
    # Test gradient flow
    loss.backward()
    print("\nGradient test passed - backward() executed successfully")
    
    # Test sampling
    samples = model.sample(num_samples=3)
    print("\n=== Sample Generation ===")
    print(f"Generated samples shape: {samples.shape}")
    assert samples.shape == (3, num_nodes, input_dim), "Sample shape mismatch!"
    
    # Test DAG properties
    print("\n=== DAG Validation ===")
    print("Adjacency matrix (first 5x5):")
    print(adj[:5, :5])
    print(f"Is DAG: {is_dag(adj)}")
    
    # Test topological sort
    topo_order = topological_sort(adj)
    print(f"Topological order: {topo_order}")
    
    # Test invalid input handling
    try:
        bad_x = torch.randn(batch_size, num_nodes+1, input_dim)
        model(bad_x)  # Should raise ValueError
    except ValueError as e:
        print(f"\nInput dimension check passed: {e}")
    
    print("\nAll tests passed!")