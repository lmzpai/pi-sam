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


class MaskDecoderLargeFeatNew(MaskDecoder):
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
            nn.Conv2d(out_st1_dim, out_st1_dim, 3, 1, 1),
            activation())

        self.mask_embedding = nn.Sequential(
            nn.Conv2d(out_st1_dim+1, out_st1_dim, 1, 1, 0), 
            LayerNorm2d(out_st1_dim),
            nn.GELU(),
            nn.Conv2d(out_st1_dim, out_st1_dim, 3, 1, 1), 
            LayerNorm2d(out_st1_dim),
            nn.GELU())
        self.output_upscaling_2 = nn.Sequential(
            nn.ConvTranspose2d(out_st1_dim, out_st1_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(out_st1_dim // 2),
            activation(),
            nn.ConvTranspose2d(out_st1_dim // 2, out_st1_dim // 4, kernel_size=2, stride=2))
        self.out_act = nn.Sequential(
            LayerNorm2d(out_st1_dim // 4),
            nn.Conv2d(out_st1_dim // 4, out_st1_dim // 4, 3, 1, 1), 
            activation())
        
        self.output_hypernetworks_mlp_2 = MLP(out_st1_dim, out_st1_dim, out_st1_dim // 4, 3)    


    def get_loss(self, preds_s, preds_l, labels):
        bs = len(preds_s)
        loss_mask_s, loss_dice_s = loss_masks(preds_s, labels, bs)
        loss_mask_l, loss_dice_l = loss_masks(preds_l, labels, bs)
        return dict(loss_mask_s=loss_mask_s, loss_dice_s=loss_dice_s,
                    loss_mask_l=loss_mask_l, loss_dice_l=loss_dice_l)


    def forward(self, encoder_outputs, interm_embeddings=None, labels=None, labels_256=None, get_loss=False):
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

        new_tokens = hyper_in
        new_src = self.embed_mask_1(up_embeddings, pred_masks_s)
        rgb_embeddings = rgb_features
        pred_masks_l = self.pred_mask_2(new_src, new_tokens, rgb_embeddings)

        outputs = dict(pred_masks = pred_masks_s, pred_masks_large = pred_masks_l)
        if get_loss:
            outputs['losses'] = self.get_loss(pred_masks_s, pred_masks_l, labels)
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
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
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
    
    def embed_mask_1(self, embedding, masks):
        return self.mask_embedding(torch.cat([embedding, masks], dim=1))

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

