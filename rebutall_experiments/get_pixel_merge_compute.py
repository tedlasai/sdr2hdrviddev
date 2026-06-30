#VMM is a per-pixel merging and doesn't fix alignment issues. We never meant to imply that VMM can be used outside this framework. Instead, we demonstrate that even a lightweight per-pixel merging (X params, X mem per frame) works well. We agree that if one were to advocate for this to be general purpose, we would require many more comparisons.

import torch
import torch.nn as nn

class Simple1x1Net(nn.Module):
    def __init__(self, in_channel=9, out_channel=9, hidden_dim=512, num_layers=3):
        super(Simple1x1Net, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channel, hidden_dim, kernel_size=1))
        layers.append(nn.LeakyReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(hidden_dim, out_channel, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


H, W = 704, 1280

model = Simple1x1Net(in_channel=9, out_channel=9, hidden_dim=64, num_layers=3)

# 1. Parameter count
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameter count: {n_params:,}")

# 2. Memory per frame at inference (float32, no gradients)
#    Input shape: (1, 9, H, W) — one frame, 3 exposures * 3 channels stacked
if torch.cuda.is_available():
    model = model.cuda()
    x = torch.zeros(1, 9, H, W, device='cuda')

    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    with torch.no_grad():
        _ = model(x)

    peak = torch.cuda.max_memory_allocated()
    mem_mb = (peak - baseline) / 1024**2
    print(f"Peak GPU memory per frame ({H}x{W}): {mem_mb:.1f} MB")
else:
    # Estimate analytically: peak activation is the hidden layer (64 channels), float32
    activation_bytes = 64 * H * W * 4
    print(f"Peak activation memory per frame ({H}x{W}): {activation_bytes / 1024**2:.1f} MB  (estimated, no CUDA)")
    input_bytes = 9 * H * W * 4
    output_bytes = 9 * H * W * 4
    print(f"  Input:  {input_bytes / 1024**2:.1f} MB")
    print(f"  Hidden: {activation_bytes / 1024**2:.1f} MB")
    print(f"  Output: {output_bytes / 1024**2:.1f} MB")
