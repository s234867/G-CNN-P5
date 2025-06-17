import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces
from escnn import nn as e2nn


class SteerableGCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_hidden,
        hidden_channels,
        max_freq=3,
        bias=True
    ):
        super().__init__()

        # Define the group space
        self.r2_act = gspaces.rot2dOnR2(N=-1)  # SO(2)
        G = self.r2_act.fibergroup

        padding = kernel_size // 2
        self.blocks = nn.ModuleList()

        # Start with trivial input type
        in_type = e2nn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        self.input_type = in_type

        # Hidden layers
        for _ in range(num_hidden):
            act = e2nn.FourierELU(
                self.r2_act,
                hidden_channels,
                irreps=G.bl_irreps(max_freq),
                N=16,
                inplace=True
            )
            out_type = act.in_type

            block = e2nn.SequentialModule(
                e2nn.R2Conv(in_type, out_type, kernel_size=kernel_size, padding=padding, bias=bias),
                e2nn.IIDBatchNorm2d(out_type),
                act
            )
            self.blocks.append(block)
            in_type = block.out_type

        # Final projection to trivial reps
        trivial_type = e2nn.FieldType(self.r2_act, hidden_channels * [self.r2_act.trivial_repr])
        self.project_to_trivial = e2nn.R2Conv(in_type, trivial_type, kernel_size=1, bias=False)
        self.invariant_map = e2nn.GroupPooling(trivial_type)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(trivial_type.size, out_channels)

    def forward(self, x):
        # Wrap input as GeometricTensor with appropriate input type
        x = e2nn.GeometricTensor(x, self.input_type)

        # Apply each equivariant block
        for block in self.blocks:
            x = block(x)

        # Project to trivial representations
        x = self.project_to_trivial(x)

        # Pool over the group dimension to get invariant features
        x = self.invariant_map(x)

        # Pool over spatial dimensions and flatten
        x = self.pool(x.tensor).squeeze()

        # Final classification layer
        return self.classifier(x)





# === Example test ===
if __name__ == "__main__":
    model = SteerableGCNN(
        in_channels=1,
        out_channels=10,
        kernel_size=5,
        num_hidden=4,
        hidden_channels=8,
        max_freq=3
    )
    x = torch.randn(4, 1, 29, 29)
    out = model(x)
    print("Output shape:", out.shape)  # should be [4, 10]
