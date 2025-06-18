# PyTorch
import torch



####################



# Baseline group H
class Group(torch.nn.Module):
    def __init__(
        self,
        dimension,
        identity
    ) -> None:
        super().__init__()

        # Save config
        self.dimension = dimension

        # Register buffers (not trainable)
        self.register_buffer(name="identity", tensor=torch.tensor(identity))

        # Save device
        self.device = self.identity.device

    def elements(self):
        """
        Group elements $H = \{ e, h, h', \dots \}$.
        """
        raise NotImplementedError()

    def product(self, h, h_prime):
        """
        Group product $h \bullet h'$.
        """
        raise NotImplementedError()
    
    def inverse(self, h):
        """
        Inverse group element $h^{-1}$.
        """
        raise NotImplementedError()
    
    def matrix_representation(self, h):
        """
        Matrix representation $\bm{D}(h) \in \mathrm{GL}_n(\mathbb{R})$.
        """
        raise NotImplementedError()
    
    def left_regular_representation(self, h, x):
        """
        Left-regular representation $\mathcal{L}^{G}_{h}$, with $h \in H$ and $x \in \mathbb{R}^2$.
        """
        raise NotImplementedError()
    
    def determinant(self, h):
        """
        Determinant of matrix representation $\mathrm{det}(\bm{D}(h))$.
        """
        raise NotImplementedError()

    def normalize_group_elements(self, h):
        """
        Normalize group elements to work with PyTorch grids in range [-1, 1].
        """
        raise NotImplementedError()


class CyclicGroup(Group):
    def __init__(
            self,
            n:int
        ) -> None:
        super().__init__(
            dimension=1,  # (θ)
            identity=[0.] # zero
        )

        # Save config
        self.n = n
        self.twopi = 2 * torch.pi

    def elements(self):
        return torch.linspace(
            start=0,
            end=self.twopi * float(self.n - 1) / float(self.n),
            steps=self.n,
            device=self.device
        )

    def product(self, h, h_prime):
        return torch.remainder(h + h_prime, self.twopi)
    
    def inverse(self, h):
        return torch.remainder(-h, self.twopi)
    
    def matrix_representation(self, h):
        cos_theta = torch.cos(h)
        sin_theta = torch.sin(h)

        return torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1), # row vector
            torch.stack([sin_theta, cos_theta], dim=-1)   # row vector
        ], dim=-2) # (B, 2, 2) if batched, else (2, 2)
    
    def left_regular_representation(self, h, x):
        D_h = self.matrix_representation(h) # (B, 2, 2) or (2, 2)

        if x.ndim == 1: # (2,) vector
            return torch.matmul(D_h, x)
        elif x.ndim == 2: # (2, W) matrix/multiple vectors
            return torch.matmul(D_h, x)
        elif x.ndim == 3: # (2, H, W) grid
            # If D_h is (B, 2, 2), then x is (2, H, W) -> output (B, 2, H, W)
            # If D_h is (2, 2), then x is (2, H, W) -> output (2, H, W)
            # the dots ... are for the potential batched dim B
            return torch.einsum("...ij,jhw->...ihw", D_h, x)
        else:
            raise ValueError(f"Unsupported input dimension for x: {x.ndim}")
    
    def determinant(self, h):
        # For rotations, determinant is always 1
        return torch.ones_like(h, device=self.device)
    
    def normalize_group_elements(self, h):
        # Normalize θ from [0, 2pi * (n-1)/n] to [-1, 1]
        return 2 * h / (self.twopi * (self.n - 1) / self.n) - 1.


class DihedralGroup(Group):
    def __init__(
            self,
            n:int
        ) -> None:
        super().__init__(
            dimension=2,      # (θ, ε)
            identity=[0., 0.] # zero, zero
        )

        # Save config
        self.n = n
        self.twopi = 2 * torch.pi
    
    def elements(self):
        # Rotations
        r = torch.linspace(
            start=0,
            end=self.twopi * float(self.n - 1) / float(self.n),
            steps=self.n,
            device=self.device
        )

        # Full set of rotations and reflections (2n)
        full_set = r.repeat(2)

        # Flags
        flags = torch.cat((
            torch.zeros(self.n, device=self.device),  # n rotations (ε = 0), r
            torch.ones(self.n, device=self.device)    # n reflections (ε = 1), rk
        ))

        return torch.stack((full_set, flags), dim=1)
    
    def product(self, h, h_prime):
        # Get theta and epsilon for both group elements (use ellipsis)
        theta_1, eps_1 = h[..., 0], h[..., 1]
        theta_2, eps_2 = h_prime[..., 0], h_prime[..., 1]

        # Compute group product
        theta_new = torch.remainder(theta_1 + (-1)**eps_1 * theta_2, self.twopi)
        eps_new = torch.remainder(eps_1 + eps_2, 2.0)

        # Stack as tuple (θ, ε)
        return torch.stack((theta_new, eps_new), dim=-1)

    def inverse(self, h):        
        theta, eps = h[..., 0], h[..., 1]

        # Inverse of theta depends on epsilon: if reflection, theta flips sign
        theta_inv = torch.where(
            condition= (eps == 0), # If not a reflection (i.e., rotation)
            input=-theta,          # Then inverse is -theta
            other=theta            # Else (it's a reflection), inverse is theta
        )
        theta_inv = torch.remainder(theta_inv, self.twopi)
        
        # Inverse of reflection is reflection itself
        eps_inv = eps

        return torch.stack((theta_inv, eps_inv), dim=-1)
    
    def matrix_representation(self, h):
        theta, eps = h[..., 0], h[..., 1]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Rotation matrix
        R_theta = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1), # row vector
            torch.stack([sin_theta, cos_theta], dim=-1)   # row vector
        ], dim=-2) # (B, 2, 2)

        # Reflection matrix (x-axis)
        S = torch.tensor([
            [1.,  0.],
            [0., -1.]
        ], device=self.device, dtype=theta.dtype)

        # Apply S * R(θ) if eps == 1, else R(θ)
        out = torch.where(
            eps[..., None, None] == 1.0, # create singleton dimension and compare
            torch.matmul(S, R_theta),    # Reflection if epsilon is 1
            R_theta                      # Otherwise, just rotation
        )

        return out
    
    def left_regular_representation(self, h, x):
        D_h = self.matrix_representation(h) # (B, 2, 2) or (2, 2)

        if x.ndim == 1: # Single vector: (2,)
            return torch.matmul(D_h, x)
        elif x.ndim == 2: # Matrix or multiple vectors: (2, W)
            return torch.matmul(D_h, x)
        elif x.ndim == 3: # Grids: (2, H, W)
            # If D_h is (B, 2, 2), then x is (2, H, W) -> output (B, 2, H, W)
            # If D_h is (2, 2), then x is (2, H, W) -> output (2, H, W)
            # the ellipsis ... are for the potential batched dim B
            return torch.einsum("...ij,jhw->...ihw", D_h, x)
        else:
            raise ValueError(f"Unsupported input dimension for x: {x.ndim}")
    
    def determinant(self, h):
        # 1 for rotation (epsilon=0), -1 for reflection (epsilon=1)
        return 1.0 - 2.0 * h[..., 1]
    
    def normalize_group_elements(self, h):
        # Correctly extract theta and epsilon for arbitrary leading dimensions
        theta = h[..., 0]
        eps = h[..., 1]
        theta_norm = 2 * theta / (self.twopi * (self.n - 1) / self.n) - 1.0
        eps_norm = 2 * eps - 1.0  # map 0 → -1 and 1 → 1
        return torch.stack((theta_norm, eps_norm), dim=-1)