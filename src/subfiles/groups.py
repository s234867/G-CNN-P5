## subfiles/groups.py ##

# Imports
import numpy as np

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

        self.dimension = dimension
        self.register_buffer(
            name='identity',
            tensor=torch.tensor(identity, dtype=torch.float32)
        ) # the identity is not a parameter
    
    # Group elements
    def elements(self):
        """
        Group elements $H = \{ e, h, h', \dots \}$.
        """
        raise NotImplementedError()
    
    # Group product
    def product(self, h, h_prime):
        """
        Group product $h \mathbin{\\bullet} h'$.
        """
        # Group product between g and g_prime
        raise NotImplementedError()
    
    # Inverse
    def inverse(self, h):
        """
        Inverse group element $h^{-1}$.
        """
        raise NotImplementedError()
    
    # Matrix representation
    def matrix_representation(self, h):
        """
        Matrix representation $\\bm{D}(h) \in \mathrm{GL}_n(\mathbb{R})$.
        """
        raise NotImplementedError()
    
    # Left-regular representation (action) on ℝ²
    def left_regular_representation(self, h, x):
        """
        Left-regular representation $\mathcal{L}^{G}_{h}$.
        """
        raise NotImplementedError()
    
    # Determinant of representation of group element
    def determinant(self, h):
        raise NotImplementedError()
    
    # Normalize group elements, useful for when we are working with normalized grids [-1, 1]
    # generally useful for PyTorch, makes distribution of data easier to handle
    def normalize_group_elements(self, h):
        raise NotImplementedError()
    


# Cyclic group C_n (optimized)
class CyclicGroup(Group):
    def __init__(self, n:int):
        super().__init__(
            dimension=1, # can be represented by 1 parameter θ, over one axis
            identity=[0.], # zero (no rotation) is the identity element of the cyclic group
        )
        
        self.n = n
        self.twopi = 2 * np.pi # faster constant
    
    def elements(self):
        return torch.linspace(
            start=0,
            end=self.twopi * float(self.n - 1) / float(self.n),
            steps=self.n,
            device=self.identity.device
        )

    def product(self, h, h_prime): # (θ + θ') mod 2π
        return torch.remainder(h + h_prime, self.twopi)
    
    def inverse(self, h): # h⁻¹ = -θ
        return torch.remainder(-h, self.twopi)
    
    def matrix_representation(self, h):
        cos_theta = torch.cos(h)
        sin_theta = torch.sin(h)

        # shape: [B, 2, 2]
        return torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta,  cos_theta], dim=-1)
        ], dim=-2)
    
    def left_regular_representation(self, h, x):
        """
        Apply rotation(s) h ∈ H to grid x ∈ ℝ².

        - h: (|H|,) or scalar
        - x: (2, H, W)

        Returns:
        - transformed: (|H|, 2, H, W) if h is batched
        """
        BATCHED = h.ndim > 0  # h: shape (|H|,) or ()

        # [2, H, W] → [H*W, 2]
        x_flat = x.view(2, -1).T  # (N, 2)

        # matrix_representation returns (|H|, 2, 2) or (2, 2)
        D_h = self.matrix_representation(h)

        if not BATCHED:
            # (2, 2) @ (N, 2).T → (2, N) → reshape
            x_trans = D_h @ x_flat.T  # (2, N)
            return x_trans.view(2, *x.shape[1:])  # (2, H, W)
        else:
            # h is batched: D_h → (|H|, 2, 2), x_flat → (1, N, 2)
            x_batched = x_flat.unsqueeze(0).expand(D_h.shape[0], -1, -1)  # (|H|, N, 2)
            x_trans = torch.bmm(x_batched, D_h.transpose(1, 2))  # (|H|, N, 2)
            x_trans = x_trans.transpose(1, 2).view(-1, 2, *x.shape[1:])  # (|H|, 2, H, W)
            return x_trans.permute(1, 0, 2, 3)  # (2, |H|, H, W)

    def determinant(self, h):
        # For rotations, determinant is always 1
        return torch.ones_like(h, device=self.identity.device)
    
    def normalize_group_elements(self, h):
        # Normalize θ from [0, 2pi * (n-1)/n] to [-1, 1]
        return 2 * h / (self.twopi * (self.n - 1) / self.n) - 1.



# Dihedral group D_n
class DihedralGroup(Group):
    def __init__(self, n: int):
        super().__init__(
            dimension=2, # theta and epsilon (rotation and reflection)
            identity=[0., 0.],  # θ=0, ε=0
        )
        self.n = n # Store as int
        self.twopi = 2 * np.pi
    
    def elements(self):
        # Rotations
        r = torch.linspace(
            start=0,
            end=self.twopi * float(self.n - 1) / float(self.n),
            steps=self.n,
            device=self.identity.device
        )

        # Full set of rotations and reflections (2*n)
        full_set = r.repeat(2)

        # Flags
        flags = torch.cat((
            torch.zeros(self.n, dtype=torch.float32, device=self.identity.device),  # n elements with no reflection (0)
            torch.ones(self.n, dtype=torch.float32, device=self.identity.device)    # n elements with reflection (1)
        ))
        
        return torch.stack((full_set, flags), dim=1) # tensor of tuples
    
    def product(self, h, h_prime):
        # If non-batched input
        if h.ndim == 1:
            h = h.unsqueeze(0)
        if h_prime.ndim == 1:
            h_prime = h_prime.unsqueeze(0)

        theta_1, eps_1 = h[:, 0], h[:, 1]
        theta_2, eps_2 = h_prime[:, 0], h_prime[:, 1]

        theta_new = torch.remainder(
            theta_1 + (-1)**eps_1 * theta_2,
            self.twopi
        )
        eps_new = torch.remainder(
            eps_1 + eps_2,
            2.0
        )

        return torch.stack((theta_new, eps_new), dim=1)
    
    def inverse(self, h):
        # If non-batched input
        if h.ndim == 1:
            h = h.unsqueeze(0)
        if h_prime.ndim == 1:
            h_prime = h_prime.unsqueeze(0)
        
        theta, eps = h[:, 0], h[:, 1]

        theta_inv = torch.where(
            condition= eps==0,
            input=-theta, # if condition is met
            other=theta   # if condition is not met
        )

        theta_inv = torch.remainder(theta_inv, self.twopi)
        eps_inv = eps # inverse of reflection is reflection; inverse reflection matrix is its own inverse

        return torch.stack((theta_inv, eps_inv), dim=1)
    
    def matrix_representation(self, h):
        # If non-batched input
        if h.ndim == 1:
            h = h.unsqueeze(0)

        theta, eps = h[:, 0], h[:, 1]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Rotation matrix
        R_theta = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta,  cos_theta], dim=-1)
        ], dim=-2)

        # Reflection matrix (x-axis)
        S = torch.tensor([
            [1.,  0.],
            [0., -1.]
        ], device=self.identity.device)

        # Apply reflection if ε == 1
        S = S.expand(h.shape[0], -1, -1)

        out = torch.where(
            eps.view(-1, 1, 1) == 1.0,
            torch.matmul(S, R_theta), # only apply reflection if condition is met (if ε=1)
            R_theta
        )

        return out
    
    def left_regular_representation(self, h, x):
        # If non-batched input
        if h.ndim == 1:
            h = h.unsqueeze(0)
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # h: [B, 2], x: [B, 2]
        D_h = self.matrix_representation(h) # [B, 2, 2]
        
        return torch.bmm(D_h, x.unsqueeze(-1)).squeeze(-1)
    
    def determinant(self, h):
        # If non-batched input
        if h.ndim == 1:
            h = h.unsqueeze(0)

        # 1 for rotation, -1 for reflection
        return 1.0 - 2.0 * h[:, 1]
    
    def normalize_group_elements(self, h):
        # If non-batched input
        if h.ndim == 1:
            h = h.unsqueeze(0)

        theta, eps = h[:, 0], h[:, 1]
        theta_norm = 2 * theta / (self.twopi * (self.n - 1) / self.n) - 1.0
        eps_norm = 2 * eps - 1.0  # map 0 → -1, 1 → 1
        return torch.stack((theta_norm, eps_norm), dim=1)



if __name__ == '__main__':
    #from utils import group_axiom_checker
    C4 = CyclicGroup(n=4)
    elements = C4.elements()
    print(elements)
    print(C4.matrix_representation(elements[1]))

    # Test left regular
    kernel_size = 4
    grid_1d = torch.linspace(
            start=-1.,
            end=1.,
            steps=kernel_size
        )
    grid_2d = torch.stack(
        torch.meshgrid(
            grid_1d, # direction i
            grid_1d, # direction j
            indexing="ij"
        )
    )
    print("grid shape:", grid_2d.shape)

    print(C4.left_regular_representation(elements[0], grid_2d))
    print(C4.left_regular_representation_batched(elements[0], grid_2d))
    
    #print(elements[1], C4.inverse(elements[1]))
    #print(torch.remainder(elements[1]+C4.inverse(elements[1]), 2*torch.pi))
    #print(group_axiom_checker(C4))

    #D4 = DihedralGroup(n=4)
    #elements = D4.elements()
    #print(elements)
    #print(elements[2], elements[6])
    #print(D4.product(elements[2], elements[6]))

    #print(D4.matrix_representation(elements[3]))
    #print(D4.left_regular_representation(elements[0], torch.tensor([1, 1])))