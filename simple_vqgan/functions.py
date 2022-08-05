# @title Loading of libraries and definitions
 
import math
import sys
import numpy as np
import os
import requests
 
# Some models include transformers, others need explicit pip install
# import transformers

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import trange
import warnings
 
import clip
import numpy as np
import imageio
import kornia.augmentation as K

normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
 
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))
 
 
def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()
 
 
def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]
 
 
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size
 
    input = input.view([n * c, 1, h, w])
 
    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])
 
    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])
 
    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)
 
 
class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
 
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)
 
 
replace_grad = ReplaceGrad.apply
 
 
class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)
 
    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None
 
 
clamp_with_grad = ClampWithGrad.apply
 
 
def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)
 
 
class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))
 
    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()
 
 
def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])
 
 
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            # K.RandomSolarize(0.01, 0.01, p=0.7),
            K.RandomSharpness(0.3,p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7))
        # self.augs = nn.Sequential(
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomAdjustSharpness(0.3,p=0.4),
        #     transforms.RandomAffine(degrees=30, translate=(0,0.1)),
        #     transforms.RandomPerspective(0.2,p=0.4),
        #     transforms.ColorJitter(hue=0.01, saturation=0.01))
        self.noise_fac = 0.1
 
 
    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = self.augs(torch.cat(cutouts, dim=0))
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch
 
 
def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model
 
 
def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

def train(pbar, opt, make_cutouts, z, z_orig, z_min, z_max, model, perceptor, init_weight, pMs, fname):
    opt.zero_grad()
    z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    out = clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
    lossAll = []
    if init_weight:
        lossAll.append(F.mse_loss(z, z_orig) * init_weight / 2)
    for prompt in pMs:
        lossAll.append(prompt(iii))
    img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
    img = np.transpose(img, (1, 2, 0))
    if fname is not None:
        imageio.imwrite(fname, np.array(img))
    loss = sum(lossAll)
    pbar.set_postfix_str(f"loss={loss.item():05f}")
    loss.backward()
    opt.step()
    with torch.no_grad():
        z.copy_(z.maximum(z_min).minimum(z_max))

def generate(
    text_prompts,
    width=250,
    height=250,
    iterations=100,
    save_every=10,
    save_steps=False,
    step_size=0.15,
    base_dir=None,
    save_name=None,
    angle=0,
    zoom=1,
    translation_x=0,
    translation_y=0,
    seed=None,
    device='cpu',
    ):

    if isinstance(text_prompts, str):
        text_prompts = [phrase.strip() for phrase in text_prompts.split("|")]
        if text_prompts == ['']:
            text_prompts = []

    if save_name is None:
        if isinstance(text_prompts, (list, tuple)):
            save_string = '__'.join(text_prompts)
        else:
            save_string = text_prompts
        save_name = '_'.join(save_string.split(' ')) + '.png'

    width = int(width)
    height = int(height)

    angle = float(angle)
    zoom = float(zoom)
    translation_x = float(translation_x)
    translation_y = float(translation_y)
    save_every = int(save_every)

    init_weight = 0
    cutn=64
    cut_pow=1.

    model="vqgan_imagenet_f16_16384"
    clip_model='ViT-B/32'

    if base_dir is None:
        warnings.warn("Warning: Because you did not specify a base_dir,\nmodels and renders will be placed within the current directory.")
        base_dir = '.'

    if not os.path.exists(base_dir+'/models'):
        os.makedirs(base_dir+'/models')
    vqgan_config=f'{base_dir}/models/{model}.yaml'
    if not os.path.exists(vqgan_config):
        print("model .yaml not found, downloading...")
        open(vqgan_config, "wb").write(requests.get('https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1').content)

    vqgan_checkpoint=f'{base_dir}/models/{model}.ckpt'
    if not os.path.exists(vqgan_checkpoint):
        print("model .ckpt not found, downloading...")
        open(vqgan_checkpoint, "wb").write(requests.get('https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1').content)

    print('Using device:', device)
    if seed is None:
        seed = torch.seed()
    torch.manual_seed(seed)
    print('Using seed', seed)
    
    model = load_vqgan_model(vqgan_config, vqgan_checkpoint).to(device)
    perceptor = clip.load(clip_model, device=device, jit=False)[0].eval().requires_grad_(False).to(device).to(torch.float32)
    
    cut_size = perceptor.visual.input_resolution
    e_dim = model.quantize.e_dim
    f = 2**(model.decoder.num_resolutions - 1)
    make_cutouts = MakeCutouts(cut_size, cutn, cut_pow=cut_pow)
    n_toks = model.quantize.n_e
    toksX, toksY = width // f, height // f
    sideX, sideY = toksX * f, toksY * f
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    one_hot = F.one_hot(
        torch.randint(n_toks, [toksY * toksX], device=device), n_toks
    ).float()
    z = one_hot @ model.quantize.embedding.weight
    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

    z_orig = z.clone()
    z.requires_grad_(True)
    opt = optim.Adam([z], lr=step_size)

    pMs = []

    for prompt in text_prompts:
        txt, weight, stop = parse_prompt(prompt)
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))

    if not os.path.exists(base_dir+'/renders'):
        os.makedirs(base_dir+'/renders')

    if save_steps:
        if not os.path.exists(base_dir+'/renders/steps'):
            os.makedirs(base_dir+'/renders/steps')

    t = trange(iterations)
    for i in t:
        if i == iterations-1:
            fname=base_dir+'/renders/'+save_name
        elif save_steps and (i % save_every == 0):
            fname=f"{base_dir}/renders/steps/{save_name[:-4]}_step_{i}_.png"
        else:
            fname=None
        train(t, opt, make_cutouts, z, z_orig, z_min, z_max, model, perceptor, init_weight, pMs, fname=fname)
    print(f"Saving final image as {os.path.abspath(os.curdir)}\{save_name}")
