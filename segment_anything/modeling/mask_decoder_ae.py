import cv2
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import numpy as np
from typing import Dict, List, Tuple, Optional
from .common import LayerNorm2d, MLP
from .mask_decoder_ori import MaskDecoder
from .transformer import TwoWayTransformer, TwoWayTransformerNew
from .loss import loss_masks, point_sample
from .prompt_encoder import PositionEmbeddingRandom
from PIL import Image


class MaskDecoderAE(MaskDecoder):
    def __init__(self, model_type, ckpt, interact=True):
        super().__init__(
            transformer_dim=256,
            transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                ),
            num_multimask_outputs=3,
            activation=nn.GELU,
            iou_head_depth= 3,
            iou_head_hidden_dim= 256,)
        assert model_type in ["vit_b","vit_l","vit_h"]

        self.load_state_dict(torch.load(ckpt))
        for n, p in self.named_parameters():
            p.requires_grad = False

        transformer_dim=256
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280}
        vit_dim = vit_dim_dict[model_type]

        out_st1_dim = 32 #
        activation = nn.GELU #
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        # for out in (256, 256)
        self.compress_vit_feat = nn.Sequential(
                                        nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim),
                                        nn.GELU(), 
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2))
        
        self.embedding_encoder = nn.Sequential(
                                        nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                                    )

        self.embedding_maskfeature = nn.Sequential(
                                        nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1), 
                                        LayerNorm2d(transformer_dim // 4),
                                        nn.GELU(),
                                        nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1))
        self.embedding_act = nn.Sequential(
            LayerNorm2d(out_st1_dim),
            nn.Conv2d(out_st1_dim, out_st1_dim, 3, 1, 1),
            activation())

        # for out in (1024, 1024)
        self.mask_aspp = MaskASPPDeformable()
        
        self.interact = interact

        self.output_upscaling_2 = nn.Sequential(
            nn.ConvTranspose2d(out_st1_dim, out_st1_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(out_st1_dim // 2),
            activation(),
            nn.ConvTranspose2d(out_st1_dim // 2, out_st1_dim // 4, kernel_size=2, stride=2))
        self.out_act = nn.Sequential(
            LayerNorm2d(out_st1_dim // 4),
            nn.Conv2d(out_st1_dim // 4, out_st1_dim // 4, 3, 1, 1), 
            LayerNorm2d(out_st1_dim // 4),
            activation(),
            nn.Conv2d(out_st1_dim // 4, 1, 3, 1, 1)) 
        self.interact_2 = None
        if interact:
            self.interact_2 = PointInteract()
            for n, p in self.named_parameters():
                if 'interact_2' in n:
                    continue
                p.requires_grad = False


    def get_loss(self, preds_s, preds_l, valid_mask, labels):
        bs = len(preds_s)
        if not self.interact:
            loss_mask_s, loss_dice_s = loss_masks(preds_s, labels, bs)
            loss_mask_l, loss_dice_l = loss_masks(preds_l, labels, bs)
            losses = dict(loss_mask_s=loss_mask_s, loss_dice_s=loss_dice_s,
                loss_mask_l=loss_mask_l, loss_dice_l=loss_dice_l)
        else:
            loss_mask_l, loss_dice_l = loss_masks(preds_l, labels, bs, valid_mask=valid_mask)
            losses = dict(loss_mask_l=loss_mask_l, loss_dice_l=loss_dice_l)
        return losses
    
    def get_size(self, tensor, dim):
        ori_size = tensor.shape[dim]
        if isinstance(ori_size, torch.Tensor):
            ori_size = ori_size.item()
        return ori_size

    def forward(self, encoder_outputs, interm_embeddings=None, labels=None, labels_256=None, get_loss=False,fg=None,bg=None,interactive=False):
        image_embeddings = encoder_outputs['image_embeddings']
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        size = self.get_size(vit_features, -1)
        transformer_dim = 256
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        rgb_features = interm_embeddings[-1]

        srcs, tokens = self.interact_1(**encoder_outputs)
        pred_masks_s, up_embeddings, hyper_in = self.pred_mask_1(srcs, tokens, hq_features, size, transformer_dim)

        new_src = self.mask_aspp(up_embeddings, pred_masks_s)

        valid_mask = None
        if interactive:
            new_src, valid_mask = self.interact_2(
                new_src, hyper_in, pred_masks_s=pred_masks_s, labels_256=labels_256, interactive=interactive, arg_fgp_coords=fg, arg_bgp_coords=bg)
        pred_masks_l = self.pred_mask_2(new_src, rgb_features)

        # return pred_masks_l

        outputs = dict(pred_masks = pred_masks_s, pred_masks_large = pred_masks_l, valid_mask = valid_mask)
        if get_loss:
            outputs['losses'] = self.get_loss(pred_masks_s, pred_masks_l, valid_mask, labels)
        return outputs


    def interact_1(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more detals."""
        # Concatenate output tokens
        bs = self.get_size(sparse_prompt_embeddings, 0)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(bs, -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, bs, dim=0)
        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        mask_token = mask_tokens_out[:, -1, :].unsqueeze(1)
        return src, mask_token

    def pred_mask_1(
            self, 
            srcs, 
            tokens, 
            hq_features, 
            size,
            transformer_dim):
        # Upscale mask embeddings and predict masks using the mask tokens
        srcs = srcs.transpose(1, 2).view(-1, transformer_dim, size, size)
        upscaled_embedding = self.output_upscaling(srcs)
        upscaled_embedding = self.embedding_act(
            self.embedding_maskfeature(upscaled_embedding) + hq_features)
        hyper_in = self.hf_mlp(tokens)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        return masks, upscaled_embedding, hyper_in

    def pred_mask_2(
            self,
            embedding,
            rgb_embedding):
        large_embedding = self.output_upscaling_2(embedding)
        large_masks = self.out_act(large_embedding + rgb_embedding)
        return large_masks


class MaskASPPDeformable(nn.Module):
    def __init__(self):
        super(MaskASPPDeformable, self).__init__()
        out_st1_dim = 32
        self.mask_embedding = nn.Sequential(
            nn.Conv2d(out_st1_dim+1, out_st1_dim, 1, 1, 0), 
            LayerNorm2d(out_st1_dim),
            nn.GELU(),
            nn.Conv2d(out_st1_dim, out_st1_dim, 3, 1, 1), 
            LayerNorm2d(out_st1_dim),
            nn.GELU())

        self.aspp1 = _ASPPModuleDeformable(out_st1_dim, out_st1_dim, 3, padding=1)
        self.aspp2 = _ASPPModuleDeformable(out_st1_dim, out_st1_dim, 7, padding=3)
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_st1_dim*3, out_st1_dim, kernel_size=1, stride=1, padding=0, bias=False),
            LayerNorm2d(out_st1_dim),
            nn.GELU())
        
    def forward(self, x, masks):
        x = self.mask_embedding(torch.cat([x, masks], dim=1))
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        return self.conv_out(torch.cat((x, x1, x2), dim=1))
    

class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super(_ASPPModuleDeformable, self).__init__()
        self.atrous_conv = nn.Sequential(
            DeformableConv2d(in_channels, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            LayerNorm2d(planes),
            nn.GELU())

    def forward(self, x):
        return self.atrous_conv(x)
    

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels,
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
        )
        return x


def get_center(masks):
    all_points = []
    for mask in masks:
        fg_points = torch.nonzero(mask)
        chosen_idx = random.randint(0, len(fg_points)-1)
        all_points.append(fg_points[chosen_idx])
    return torch.stack(all_points, dim=0)


class PointInteract(nn.Module):
    def __init__(self, max_points=10, min_area_thr=2):
        super().__init__()
        transformer_dim=32
        self.points_flag = nn.Embedding(2, 32)

        self.pe_layer = PositionEmbeddingRandom(transformer_dim // 2)
        self.token_embed = nn.Sequential(
            nn.LayerNorm(transformer_dim*3),
            MLP(transformer_dim*3, transformer_dim*8, transformer_dim, 3),
            nn.LayerNorm(transformer_dim))
        self.transformer=TwoWayTransformerNew(
                depth=2,
                embedding_dim=transformer_dim,
                mlp_dim=transformer_dim,
                num_heads=1,
                attention_downsample_rate=1)
        self.out_cov = nn.Conv2d(transformer_dim, transformer_dim, 3, 1, 1)
        nn.init.constant_(self.out_cov.weight, 0.)
        nn.init.constant_(self.out_cov.bias, 0.)
                
        self.max_points = max_points
        self.min_area_thr = min_area_thr
        self.hw_size = 256
        img_pe = self.pe_layer([self.hw_size, self.hw_size])
        self.register_buffer('img_pe', img_pe)
        self.bnd_op = BlockTargetGenerator()


    def get_areas(self, mask, device, flag):
        res_regions = torch.zeros((0, self.hw_size, self.hw_size), dtype=torch.bool, device=device)
        res_areas = torch.zeros((0, ), dtype=torch.int64, device=device)
        res_flags = torch.zeros((0, ), dtype=torch.int64, device=device)
        if mask.any():
            components, labels = cv2.connectedComponents(mask.astype(np.uint8))
            labels_torch = torch.tensor(labels, device=device).unsqueeze(0)
            ids_torch = torch.range(1, components-1, device=device).view(-1, 1, 1)
            regions = labels_torch == ids_torch
            areas = regions.flatten(1).sum(-1)
            valid_ind = areas >= self.min_area_thr
            if valid_ind.any():
                res_regions = regions[valid_ind]
                res_areas = areas[valid_ind]
                res_flags = torch.zeros((len(res_areas), ), dtype=torch.int64, device=device)
                res_flags += flag
        return res_regions, res_areas, res_flags

    def sample_from_areas(self, fg_areas, bg_areas, fg_masks, bg_masks, fg_flags, bg_flags):
        if (len(fg_areas) == 0) and (len(bg_areas) == 0):
            return None, None
        areas = torch.cat([fg_areas, bg_areas], dim=0)
        masks = torch.cat([fg_masks, bg_masks], dim=0)
        flags = torch.cat([fg_flags, bg_flags], dim=0)
        all_num = len(masks)
        if self.training:
            points_num = random.randint(1, min(self.max_points, all_num))
        else:
            points_num = 1
        _, chosen_inds = torch.topk(areas, k=points_num)
        chosen_masks = masks[chosen_inds]
        point_coors = get_center(chosen_masks)
        chosen_flags = flags[chosen_inds]
        return point_coors.float() / self.hw_size, chosen_flags
    
    @torch.no_grad()
    def get_points_coor(self, masks, labels, threshold = 0.5):
        if labels is None:
            labels = torch.zeros_like(masks)
        bs = masks.shape[0]
        out_masks = torch.sigmoid(masks)
        binery_out_masks = (out_masks > threshold).float()
        labels_01 = labels / 255.0
        difference = labels_01 - binery_out_masks
        fg_region=torch.zeros_like(difference)
        bg_region=torch.zeros_like(difference)
        fg_region[difference == 1] = 1
        bg_region[difference == -1] = 1
        fg_region = self.bnd_op(fg_region)
        bg_region = self.bnd_op(bg_region)
        fg_region = fg_region.cpu().numpy()
        bg_region = bg_region.cpu().numpy()
        attention_mask = torch.zeros((bs, self.max_points + 1), dtype=torch.bool, device = masks.device)
        valid_mask = torch.zeros((bs,), dtype=torch.float32, device = masks.device)
        all_points = torch.zeros((bs, self.max_points, 2), dtype=torch.float32, device = masks.device)
        all_flags = torch.zeros((bs, self.max_points), dtype=torch.int64, device = masks.device)

        for b_i in range(bs):
            fg_masks, fg_areas, fg_flags = self.get_areas(fg_region[b_i, 0], masks.device, 1)
            bg_masks, bg_areas, bg_flags = self.get_areas(bg_region[b_i, 0], masks.device, 0)
            point_coors, flags = self.sample_from_areas(fg_areas, bg_areas, fg_masks, bg_masks, fg_flags, bg_flags)
            if point_coors is not None:
                valid_mask[b_i] = 1
                num = len(point_coors)
                all_points[b_i][:num] = point_coors
                attention_mask[b_i][(num+1):] = True
                all_flags[b_i][:num] = flags
        return all_points, attention_mask, valid_mask, all_flags

    def forward(
        self,
        up_embeddings,
        hyper_in,
        pred_masks_s = None,
        labels_256 = None,
        interactive = False,
        arg_fgp_coords:Optional[torch.Tensor] = None,
        arg_bgp_coords:Optional[torch.Tensor] = None,        
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        #torch.Size([1, 10, 2]) torch.Size([1, 11]) torch.Size([1]) torch.Size([1, 10])
        print(interactive)
        if not interactive:
            all_points, attn_mask, valid_mask, all_flags = self.get_points_coor(pred_masks_s, labels_256)
        else:
            all_points = torch.cat([arg_fgp_coords, arg_bgp_coords], dim=1).float()
            all_flags = torch.cat([torch.ones(arg_fgp_coords.shape[1], device=all_points.device), torch.zeros(arg_bgp_coords.shape[1], device=all_points.device)], dim=0).unsqueeze(0).long()
            valid_mask = torch.ones(all_points.shape[0], dtype=torch.float32, device=all_points.device)
            attn_mask = torch.zeros((all_points.shape[0], all_points.shape[1] + 1), dtype=torch.bool, device=all_points.device)
        print(all_points.shape, attn_mask.shape, valid_mask.shape, all_flags.shape)
        # vis_points(all_points, attn_mask, all_flags, pred_masks_s, labels_256, valid_mask)

        points_feat = point_sample(up_embeddings, all_points).transpose(1, 2).contiguous()
        points_flag_embedding = self.points_flag(all_flags)
        points_pe = self.pe_layer.forward_with_coords_norm(all_points)

        points_token = self.token_embed(torch.cat([points_feat, points_flag_embedding, points_pe], dim=-1))
        # points_token = self.token_embed(points_feat + points_flag_embedding + points_pe)
        token = torch.cat([hyper_in, points_token], dim=1)

        bs = hyper_in.size(0)
        img_pe = self.img_pe.unsqueeze(0).expand(bs, -1, -1, -1)

        src = self.transformer(up_embeddings, img_pe, token, kv_mask=attn_mask)
        src = src.view(bs, self.hw_size, self.hw_size, -1).permute(0, 3, 1, 2).contiguous()
        return self.out_cov(src)+up_embeddings, valid_mask
        

class BlockTargetGenerator(nn.Module):
    def __init__(self, boundary_width=1):
        super(BlockTargetGenerator, self).__init__()
        self.boundary_width = boundary_width
        self.kernel_size = 2 * boundary_width + 1

        # Laplacian kernel
        laplacian_kernel = -torch.ones(1, 1, self.kernel_size, self.kernel_size).to(dtype=torch.float32).requires_grad_(False)
        laplacian_kernel[0, 0, boundary_width, boundary_width] = self.kernel_size ** 2 - 1
        self.laplacian_kernel = nn.Parameter(laplacian_kernel, requires_grad=False)

    def forward(self, mask_target):
        mask_target = mask_target.float()

        # Pad target
        pad_target = F.pad(mask_target, (self.boundary_width, self.boundary_width, self.boundary_width, self.boundary_width), "constant", 0)

        # Positive boundary
        pos_boundary_targets = F.conv2d(pad_target, self.laplacian_kernel, padding=0)
        pos_boundary_targets = pos_boundary_targets.clamp(min=0) / float(self.kernel_size ** 2)
        pos_boundary_targets[pos_boundary_targets > 0.1] = 1
        pos_boundary_targets[pos_boundary_targets <= 0.1] = 0
        pos_boundary_targets = pos_boundary_targets

        # Generate block target
        foreground_inds = (mask_target - pos_boundary_targets) > 0

        return foreground_inds
    

def vis_points(all_points, attn_mask, all_flags, pred_masks_s, labels_256, valid_mask):
    idx = 0
    for (point, attn, flag, mask_p, label_p, valid) in zip(all_points, attn_mask, all_flags, pred_masks_s, labels_256, valid_mask):
        attn = attn[1:]
        mask_p = mask_p[0]
        label_p = label_p[0]
        point = point[~attn]
        point = (point * 255).cpu().numpy().astype(np.int64)
        flag = flag[~attn]
        mask_p = (mask_p > 0).detach().cpu().numpy().astype(np.uint8)
        label_p = label_p.detach().cpu().numpy().astype(np.uint8)
        mask_p = np.stack([mask_p, mask_p, mask_p], axis=-1)
        mask_p = mask_p * 255
        label_p = np.stack([label_p, label_p, label_p], axis=-1)
        label_p = label_p * 255
        mask_p = visualize_points_cv2(mask_p, point, flag)
        out = np.concatenate([mask_p, label_p], axis=1)
        Image.fromarray(out).save(f"tmp/mask_{idx}.png")
        idx += 1



def visualize_points_cv2(image, coordinates, flag):
    """
    Visualize points on the image using OpenCV.
    
    Args:
    - image: numpy array, shape (h, w, 3), input image.
    - coordinates: numpy array, shape (n, 2), coordinates of points to visualize.
    
    Returns:
    - image_with_points: numpy array, shape (h, w, 3), image with points visualized.
    """
    # Draw circles for each coordinate
    for coord, f in zip(coordinates, flag):
        if f == 1:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        cv2.circle(image, (tuple(coord)[1], tuple(coord)[0]), 5, color, -1)  # Draw a filled circle
    
    return image
