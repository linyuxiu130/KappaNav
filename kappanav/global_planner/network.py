"""One DINOv3 pass, one same-grid fusion conditioned on ground resolution, one local native-pixel readout of the surface
classes."""
import torch
from torch import nn
from torch.nn import functional as F

from kappanav.global_planner.taxonomy import CLASSES


def local_upsample(logits, kernels):
    """UPLiFT-style local pooling; shared sum-one weights let us classify before upsampling."""
    b, c, h, w = logits.shape
    with torch.autocast(logits.device.type, enabled=False):
        values = F.unfold(F.pad(logits.float(), (1, 1, 1, 1), mode='replicate'), 3)
        values = values.reshape(b, c, 9, h, w)
        weights = kernels.float().reshape(b, 9, 16, h, w).softmax(1)
        pixels = torch.einsum('bcnhw,bnshw->bcshw', values, weights)
        return F.pixel_shuffle(pixels.flatten(1, 2), 4)


class MaskedSemanticReadout(nn.Conv2d):
    """One class-masked context read, initialized to the original pixel classifier."""
    def __init__(self, channels):
        super().__init__(channels, len(CLASSES), 1)
        self.context_weight = nn.Parameter(torch.zeros(channels, channels))

    def forward(self, x):
        scores = super().forward(x)
        with torch.autocast(x.device.type, enabled=False):
            flat = scores.float().flatten(2)
            mask = (flat < 0).detach()
            # Mask2Former unmasks an empty predicted region to keep its attention defined.
            mask = mask & ~mask.all(-1, keepdim=True)
            attention = (flat / x.shape[1]**.5).masked_fill(mask, -torch.inf).softmax(-1)
            values = x.float().flatten(2)
            context = attention @ values.transpose(1, 2)
            delta = F.linear(context, self.context_weight)
            return scores.float() + (delta @ values).reshape_as(scores)


class DenseDecoder(nn.Module):
    """Fuse encoder layers once, scale them by the ground resolution, inject aligned RGB detail once; native logits
    of the surface classes."""
    def __init__(self, feature_dim, reference_gsd):
        super().__init__()
        self.reference_gsd = float(reference_gsd)
        width = 128
        self.projections = nn.ModuleList(nn.Conv2d(feature_dim, width, 1) for _ in range(4))
        self.fusion = nn.Sequential(
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width, 1))
        # An exact rearrangement preserves every RGB pixel and its subpixel phase.
        self.detail = nn.Sequential(
            nn.PixelUnshuffle(4), nn.Conv2d(3*16, 32, 1), nn.GELU())
        self.refine = nn.Sequential(
            nn.Conv2d(width+32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.classifier = MaskedSemanticReadout(64)
        self.kernels = nn.Conv2d(64, 9*16, 1)
        self.residual_scale = nn.Parameter(torch.zeros(()))
        # The metres per pixel tell a blade of grass from a hedge of the same texture; zero-initialised.
        self.resolution = nn.Linear(1, 2 * width)
        nn.init.zeros_(self.resolution.weight)
        nn.init.zeros_(self.resolution.bias)

    def forward(self, features, rgb, gsd):
        x = sum(F.gelu(project(feature)) for project, feature in zip(self.projections, features)) / 4
        x = x + self.residual_scale * self.fusion(x)
        scale, shift = self.resolution(torch.log(gsd / self.reference_gsd).reshape(-1, 1).to(x.dtype)).chunk(2, 1)
        x = x * (1 + scale[..., None, None]) + shift[..., None, None]
        detail = self.detail(rgb)
        # Patch centres and RGB 4x4 cells share the same image extent.
        x = F.interpolate(x, size=detail.shape[-2:], mode='bilinear', align_corners=False)
        x = self.refine(torch.cat((x, detail), 1))
        return local_upsample(self.classifier(x), self.kernels(x))


class GlobePlanner(nn.Module):
    def __init__(self, backbone, reference_gsd):
        super().__init__()
        self.backbone = backbone.eval()
        for name, parameter in self.backbone.named_parameters():
            # The unused SSL mask token has no supervised forward path; keep it out of DDP gradients.
            parameter.requires_grad_(name != 'embeddings.mask_token')
        self.indices = (5, 11, 17, 23)
        self.head = DenseDecoder(backbone.config.hidden_size, reference_gsd)

    def train(self, mode=True):
        super().train(mode)
        # Keep DINOv3's self-supervised coordinate perturbations disabled; its layer weights still learn.
        self.backbone.eval()
        return self

    def forward(self, rgb, gsd):
        """rgb normalised B x 3 x H x W, gsd metres per pixel B -> class logits B x K x H x W."""
        b, _, h, w = rgb.shape
        patch = self.backbone.config.patch_size
        if h % patch or w % patch:
            raise ValueError('Input sides must be multiples of the patch size')
        features = []
        rope = self.backbone.rope_embeddings(rgb)
        tokens = self.backbone.embeddings(rgb)
        for index, block in enumerate(self.backbone.layer):
            tokens = block(tokens, attention_mask=None, position_embeddings=rope)
            if index in self.indices:
                value = self.backbone.norm(tokens[:, -(h//patch)*(w//patch):])
                features.append(value.transpose(1, 2).reshape(b, -1, h//patch, w//patch).contiguous())
        return self.head(features, rgb, gsd)
