import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from .common import LayerNorm2d, MLP
from .mask_decoder_ori import MaskDecoder
from .transformer import TwoWayTransformer
from .loss import loss_masks, point_sample
from .prompt_encoder import PositionEmbeddingRandom


class MaskDecoderPi(MaskDecoder):
    def __init__(self, model_type, ckpt):
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
        #the following code is different from the original MaskDecoderHQ
        self.embedding_act = nn.Sequential(
            LayerNorm2d(out_st1_dim),
            activation())

        self.interact_2 = PointInteract()

        self.output_upscaling_2 = nn.Sequential(
            nn.ConvTranspose2d(out_st1_dim, out_st1_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(out_st1_dim // 2),
            activation(),
            nn.ConvTranspose2d(out_st1_dim // 2, out_st1_dim // 4, kernel_size=2, stride=2))
        self.out_act = nn.Sequential(
            LayerNorm2d(out_st1_dim // 4),
            activation())
        
        self.output_hypernetworks_mlp_2 = MLP(out_st1_dim, out_st1_dim, out_st1_dim // 4, 3)    


    def get_loss(self, preds_s, preds_l, preds_pi, valid_mask, labels):
        bs = len(preds_s)
        loss_mask_s, loss_dice_s = loss_masks(preds_s, labels, bs)
        loss_mask_l, loss_dice_l = loss_masks(preds_l, labels, bs)
        loss_mask_pi, loss_dice_pi = loss_masks(preds_pi, labels, bs, valid_mask=valid_mask)
        return dict(loss_mask_s=loss_mask_s, loss_dice_s=loss_dice_s,
                    loss_mask_l=loss_mask_l, loss_dice_l=loss_dice_l,
                    loss_mask_pi=loss_mask_pi, loss_dice_pi=loss_dice_pi)


    def forward(self, encoder_outputs, interm_embeddings=None, labels=None, labels_256=None, get_loss=False,fg=None,bg=None,interactive=False):
        image_embeddings = torch.cat([enc_out['image_embeddings'] for enc_out in encoder_outputs], dim=0)
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        size = vit_features.shape[-1]
        transformer_dim = 256
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)
        rgb_features = interm_embeddings[-1]
        srcs, tokens = [], []
        bs = len(encoder_outputs)
        for b_i, enc_out in enumerate(encoder_outputs):
            src, token = self.interact_1(**enc_out)
            srcs.append(src)
            tokens.append(token)
        srcs = torch.cat(srcs, dim=0)
        tokens = torch.stack(tokens, dim=0)
        pred_masks_s, up_embeddings, hyper_in = self.pred_mask_1(srcs, tokens, hq_features, size, transformer_dim)

        new_tokens, new_src, valid_mask = self.interact_2(
            up_embeddings, hyper_in, pred_masks_s=pred_masks_s, labels_256=labels_256, interactive=interactive, arg_fgp_coords=fg, arg_bgp_coords=bg)
        new_tokens = torch.cat([hyper_in, new_tokens], dim=0)
        new_src = torch.cat([up_embeddings, new_src], dim=0)
        rgb_features_clone = rgb_features.clone()
        rgb_embeddings = torch.cat([rgb_features, rgb_features_clone], dim=0)
        rgb_embeddings = rgb_features
        pred_masks_l = self.pred_mask_2(new_src, new_tokens, rgb_embeddings)
        pred_masks_large = pred_masks_l[:bs]
        pred_masks_pi = pred_masks_l[bs:]

        outputs = dict(pred_masks = pred_masks_s, pred_masks_large = pred_masks_large, pred_masks_pi=pred_masks_pi)
        if get_loss:
            outputs['losses'] = self.get_loss(pred_masks_s, pred_masks_large, pred_masks_pi, valid_mask, labels)
        return outputs


    def interact_1(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0).item(), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0].item(), dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0].item(), dim=0)
        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        mask_token = mask_tokens_out[:, -1, :]
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
            hyper_in,
            rgb_embedding):
        large_embedding = self.output_upscaling_2(embedding)
        large_embedding = self.out_act(large_embedding + rgb_embedding)
        large_hyper_in = self.output_hypernetworks_mlp_2(hyper_in)
        b, c, h, w = large_embedding.shape
        large_masks = (large_hyper_in @ large_embedding.view(b, c, h * w)).view(b, -1, h, w)
        return large_masks


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
            nn.LayerNorm(transformer_dim),
            MLP(transformer_dim, transformer_dim*4, transformer_dim, 3),
            nn.LayerNorm(transformer_dim))
        
        self.transformer=TwoWayTransformer(
                depth=1,
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=1,
                activation=nn.GELU,
                attention_downsample_rate=1)
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
        points_num = random.randint(1, min(self.max_points, all_num))
        _, chosen_inds = torch.topk(areas, k=points_num)
        chosen_masks = masks[chosen_inds]
        point_coors = get_center(chosen_masks)
        return point_coors.float() / self.hw_size, flags[chosen_inds]
    
    @torch.no_grad()
    def get_points_coor(self, masks, labels, threshold = 0.5):
        if labels is None:
            labels = torch.zeros_like(masks)
        bs = masks.shape[0].item()
        out_masks = torch.sigmoid(masks)
        binery_out_masks = (out_masks > threshold).float()
        labels /= 255.0
        difference = labels - binery_out_masks
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

        attn_mask, valid_mask = None, None
        if not interactive:
            all_points, attn_mask, valid_mask, all_flags = self.get_points_coor(pred_masks_s, labels_256)
            #rint(all_points.shape, attn_mask.shape, valid_mask.shape, all_flags.shape,sep='\n')
        else:
            all_points = torch.cat([arg_fgp_coords, arg_bgp_coords], dim=1).float()
            all_flags = torch.cat([torch.ones(arg_fgp_coords.shape[1], device=all_points.device), torch.zeros(arg_bgp_coords.shape[1], device=all_points.device)], dim=0).unsqueeze(0).long()
            valid_mask = torch.ones(all_points.shape[0], dtype=torch.float32, device=all_points.device)
            attn_mask = torch.zeros((all_points.shape[0],all_points.shape[1] + 1), dtype=torch.bool, device=all_points.device)
            #print(all_points.shape, attn_mask.shape, valid_mask.shape, all_flags.shape,sep='\n')

        points_feat = point_sample(up_embeddings, all_points).transpose(1, 2).contiguous()
        points_flag_embedding = self.points_flag(all_flags)
        points_pe = self.pe_layer.forward_with_coords_norm(all_points)

        points_token = self.token_embed(points_feat + points_flag_embedding + points_pe)
        token = torch.cat([hyper_in, points_token], dim=1)

        bs = hyper_in.size(0)
        img_pe = self.img_pe.unsqueeze(0).expand(bs, -1, -1, -1)

        new_token, src = self.transformer(up_embeddings, img_pe, token, kv_mask=attn_mask)
        out_token = new_token[:, 0, :].unsqueeze(1)
        src = src.view(bs, self.hw_size, self.hw_size, -1).permute(0, 3, 1, 2).contiguous()
        return out_token, src, valid_mask
        


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