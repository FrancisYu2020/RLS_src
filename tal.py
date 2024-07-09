import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBackbone(nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x

class SimpleFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(SimpleFPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            self.lateral_convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            self.output_convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        c2, c3, c4 = x
        p4 = self.lateral_convs[2](c4)
        p3 = self.lateral_convs[1](c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral_convs[0](c2) + F.interpolate(p3, scale_factor=2, mode="nearest")

        p2 = self.output_convs[0](p2)
        p3 = self.output_convs[1](p3)
        p4 = self.output_convs[2](p4)
        
        return p2, p3, p4

class SaliencyRefinement(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SaliencyRefinement, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))

class SimpleTemporalActionLocalization(nn.Module):
    def __init__(self):
        super(SimpleTemporalActionLocalization, self).__init__()
        self.backbone = SimpleBackbone()
        self.fpn = SimpleFPN([128, 256, 512], 256)
        self.boundary_refinement = SaliencyRefinement(256, 1)

    def forward(self, x):
        c1 = self.backbone.conv1(x)
        c2 = self.backbone.conv2(c1)
        c3 = self.backbone.conv3(c2)
        c4 = self.backbone.conv4(c3)

        p2, p3, p4 = self.fpn((c2, c3, c4))

        boundary_predictions = [self.boundary_refinement(p) for p in (p2, p3, p4)]
        return boundary_predictions

# Example usage
if __name__ == "__main__":
    model = SimpleTemporalActionLocalization()
    x = torch.randn(1, 3, 256)  # Example input tensor
    boundary_predictions = model(x)

    for i, bp in enumerate(boundary_predictions):
        print(f"Boundary Predictions Level {i+1} shape: {bp.shape}")
