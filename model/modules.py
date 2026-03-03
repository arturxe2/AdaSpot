# Global imports
import abc
import torch
import torch.nn as nn
from timm import create_model
import numpy as np
import torch.nn.functional as F
import math

# Local imports

class ABCModel:

    @abc.abstractmethod
    def get_optimizer(self, opt_args):
        raise NotImplementedError()

    @abc.abstractmethod
    def epoch(self, loader, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, seq):
        raise NotImplementedError()

    @abc.abstractmethod
    def state_dict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, state_dict):
        raise NotImplementedError()

class BaseRGBModel(ABCModel):

    def get_optimizer(self, opt_args):

            optimizer = torch.optim.AdamW(self._get_params(), **opt_args)

            return optimizer, \
                torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    """ Assume there is a self._model """

    def _get_params(self):
        return list(self._model.parameters())

    def state_dict(self):
        if isinstance(self._model, nn.DataParallel):
            return self._model.module.state_dict()
        return self._model.state_dict()

    def load(self, state_dict, strict = True):
        if isinstance(self._model, nn.DataParallel):
            self._model.module.load_state_dict(state_dict, strict=strict)
        else:
            self._model.load_state_dict(state_dict, strict=strict)

class CustomRegNetY(nn.Module):
    def __init__(self, feature_arch='rny002'):
        super().__init__()

        base = create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny006': 'regnety_006',
                    'rny008': 'regnety_008',
                }[feature_arch.rsplit('_', 1)[0]], pretrained=True)

        # Keep reference to original blocks
        self.stem = base.stem
        self.s1 = base.s1
        self.s2 = base.s2
        self.s3 = base.s3
        self.s4 = base.s4
        self.head = base.head

        # Get feature dimensions from the last conv layer of each stage
        d = []
        d.append(list(self.s1.children())[-1].conv1.out_channels)
        d.append(list(self.s2.children())[-1].conv1.out_channels)
        d.append(list(self.s3.children())[-1].conv1.out_channels)
        d.append(list(self.s4.children())[-1].conv1.out_channels)

        self.ds = d

    def forward(self, x, return_last_layer=False):

        x = self.stem(x)

        # Block 1
        x = self.s1(x)

        # Block 2
        x = self.s2(x)

        # Block 3
        x = self.s3(x)

        # Block 4
        x = self.s4(x)

        out = self.head(x)

        if return_last_layer:
            return out, x
        
        return out
    
class FCLayers(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(self.dropout(x).reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)
    
class MultFCLayers(nn.Module):

    def __init__(self, feat_dim, elements):
        super().__init__()
        elements = [1] + elements
        self._fc_out = nn.ModuleList([nn.Linear(feat_dim, elem) for elem in elements])
        self.dropout = nn.Dropout()
    
    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        outputs = []
        for fc in self._fc_out:
            out = fc(self.dropout(x).reshape(batch_size * clip_len, -1)).view(
                batch_size, clip_len, -1)
            outputs.append(out)
        return outputs

class ROISelector(nn.Module):
    def __init__(self, roi_size = (112, 112), spatial_increase = 8, threshold = 0.3, original_size = (448, 448)):
        super().__init__()
        self.roi_size = roi_size
        self.spatial_increase = spatial_increase
        self.threshold = threshold
        self.original_size = original_size
        
        self.sizes = (np.arange(roi_size[0], original_size[0], 28), np.arange(roi_size[1], original_size[1], 28))
    
    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, T, C, H, W)

        Returns:
            size: Tensor of shape (B, 1, T, 2) with height/width
            centers: Tensor of shape (B, 1, T, 2) with (y, x) positions
        """
        B, T, C, H, W = x.shape
        H_up, W_up = H * self.spatial_increase, W * self.spatial_increase        

        # ---- Step 1: Reduce channel dimension ----
        x = x.mean(dim=2, keepdim=True)        # -> (B, T, 1, H, W)

        # Min-max normalize (for more clear peaks, and standarized threshold)
        x = x.view(B, T, -1)  # (B, T, H*W)
        min_val, _ = x.min(dim=-1, keepdim=True)
        max_val, _ = x.max(dim=-1, keepdim=True)
        x = (x - min_val) / (max_val - min_val + 1e-5)
        x = x.view(B, T, 1, H, W)  # (B, T, 1, H, W)

        # ---- Step 2: Upsample to target spatial resolution ----
        x = x.repeat_interleave(self.spatial_increase, dim=-2)
        x = x.repeat_interleave(self.spatial_increase, dim=-1)  # -> (B, T, 1, H_up, W_up)
        x = x.permute(0, 2, 1, 3, 4)                            # -> (B, 1, T, H_up, W_up)

        # ---- Step 3: Smooth with 3D Gaussian kernel ----
        kernel_size = (7, self.spatial_increase + 1, self.spatial_increase  + 1)
        sigma = tuple(k / 4.0 for k in kernel_size)  # sigma = 1/4 of kernel size
        for _ in range(2):  # apply twice for stronger smoothing
            x = self.gaussian_pool3d(x, kernel_size=kernel_size, stride=1, sigma=sigma, pad_mode='reflect')

        # Initialize outputs
        center = torch.zeros((B, 1, T), device=x.device)
        size_h = torch.zeros((B, 1, T), device=x.device)
        size_w = torch.zeros((B, 1, T), device=x.device)

        # Re-normalize (H, W) to sum to 1
        x = x / (x.sum(dim=(-1, -2), keepdim=True) + 1e-5)

        # ---- Step 4: Search for peaks at multiple sizes ----
        for sh, sw in zip(self.sizes[0], self.sizes[1]):

            # Stop early if all frames already have assigned size
            if (size_h == 0).sum() == 0:
                break

            # Convert target sizes into upsampled coordinates
            ksh = int(sh / self.original_size[0] * H_up)
            ksw = int(sw / self.original_size[1] * W_up)

            # Ensure odd kernel sizes for pooling
            if ksh % 2 == 0: ksh += 1
            if ksw % 2 == 0: ksw += 1
            kernel_size = (1, ksh, ksw)

            # Apply avg pooling and scale result
            pooled = F.avg_pool3d(
                x, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)
            )
            pooled = pooled * math.prod(kernel_size)

            # Keep only local maxima
            max_mask = pooled == pooled.view(B, 1, T, -1).max(dim=-1, keepdim=True)[0].view(B, 1, T, 1, 1)
            pooled = pooled * max_mask.float()

            # Identify peak values and locations
            out_vals, out_idxs = pooled.view(B, 1, T, -1).max(dim=-1)
            exists = (out_vals > self.threshold) & (size_h == 0)

            # Update results where peaks exist
            center[exists] = out_idxs[exists].float()
            size_h[exists] = sh
            size_w[exists] = sw

        # ---- Step 5: Defaults for missing peaks ----
        center[center == 0] = H * W // 2                      # fallback: center
        size_h[size_h == 0] = self.sizes[0][-1]               # fallback: largest size
        size_w[size_w == 0] = self.sizes[1][-1]
        center = center.squeeze(1)
        size_h = size_h.squeeze(1)
        size_w = size_w.squeeze(1)

        sizes = torch.stack((size_h, size_w), dim=-1)            # (B, T, 2)

        centers = torch.zeros((B, T, 2), device=x.device)
        centers[..., 0] = center // W_up   # y-coordinate
        centers[..., 1] = center % W_up    # x-coordinate
        centers[..., 0] /= H_up
        centers[..., 1] /= W_up

        return centers, sizes

            
    def gaussian_kernel_3d(self, kernel_size, sigma):
        """
        Create a 3D Gaussian kernel.
        kernel_size: tuple of 3 ints (kT, kH, kW)
        sigma: tuple of 3 floats (sT, sH, sW)
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(sigma, (int, float)):
            sigma = (sigma,) * 3
                
        kT, kH, kW = kernel_size
        sT, sH, sW = sigma
                
        # Create grid
        t = torch.arange(kT) - (kT - 1) / 2
        h = torch.arange(kH) - (kH - 1) / 2
        w = torch.arange(kW) - (kW - 1) / 2
        T, H, W = torch.meshgrid(t, h, w, indexing="ij")
                
        kernel = torch.exp(-((T**2)/(2*sT**2) + (H**2)/(2*sH**2) + (W**2)/(2*sW**2)))
        kernel = kernel / kernel.sum()
        return kernel


    def gaussian_pool3d(self, x, kernel_size=3, stride=1, sigma=1.0, pad_mode="zero"):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # Build Gaussian kernel
        kernel = self.gaussian_kernel_3d(kernel_size, sigma).to(x.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,kT,kH,kW]
        kernel = kernel.repeat(x.size(1), 1, 1, 1, 1)  # [C,1,kT,kH,kW]

        padding = tuple(k // 2 for k in kernel_size)  # (t, h, w)

        if pad_mode == "zero":
            out = F.conv3d(x, kernel, stride=stride, padding=padding, groups=x.size(1))
        else:
            # Manual padding first
            if pad_mode == "reflect":
                pad_layer = nn.ReflectionPad3d((padding[2], padding[2],
                                                padding[1], padding[1],
                                                padding[0], padding[0]))
            elif pad_mode == "replicate":
                pad_layer = nn.ReplicationPad3d((padding[2], padding[2],
                                                padding[1], padding[1],
                                                padding[0], padding[0]))
            else:
                raise ValueError(f"Unknown pad_mode: {pad_mode}")
            
            x_padded = pad_layer(x)
            out = F.conv3d(x_padded, kernel, stride=stride, padding=0, groups=x.size(1))

        return out

def step(model, optimizer, scaler, loss, lr_scheduler=None):
    scaler = None
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    if lr_scheduler is not None:
        lr_scheduler.step()
    optimizer.zero_grad()