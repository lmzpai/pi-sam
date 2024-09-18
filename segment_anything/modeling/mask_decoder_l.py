import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from .common import LayerNorm2d, MLP
from .mask_decoder_ori import MaskDecoder
from .transformer import TwoWayTransformer
from .loss import loss_masks


class MaskDecoderLarge(MaskDecoder):
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
        out_st1_dim = 32
        activation = nn.GELU
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.output_upscaling_2 = nn.Sequential(
            LayerNorm2d(out_st1_dim),
            nn.ConvTranspose2d(out_st1_dim, out_st1_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(out_st1_dim // 2),
            activation(),
            nn.ConvTranspose2d(out_st1_dim // 2, out_st1_dim // 4, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlp_2 = MLP(out_st1_dim, out_st1_dim, out_st1_dim // 4, 3)    
    
    def forward(self, encoder_outputs, interm_embeddings=None, labels=None, get_loss=False):
        pred_masks_s, pred_masks_l = [], []
        for enc_out in encoder_outputs:
            mask_s, mask_l = self.forward_sample(**enc_out, multimask_output=False)
            pred_masks_s.append(mask_s)
            pred_masks_l.append(mask_l)
        pred_masks_s = torch.cat(pred_masks_s, dim=0)
        pred_masks_l = torch.cat(pred_masks_l, dim=0)
        outputs = dict(pred_masks = pred_masks_s, pred_masks_large = pred_masks_l)
        if get_loss:
            outputs['losses'] = self.get_loss(pred_masks_s, pred_masks_l, labels)
        return outputs

    def get_loss(self, preds_s, preds_l, labels):
        bs = len(preds_s)
        loss_mask_s, loss_dice_s = loss_masks(preds_s, labels, bs)
        loss_mask_l, loss_dice_l = loss_masks(preds_l, labels, bs)
        return dict(loss_mask_s=loss_mask_s, loss_dice_s=loss_dice_s,
                    loss_mask_l=loss_mask_l, loss_dice_l=loss_dice_l)

    def forward_sample(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        return self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings)


    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in = self.hf_mlp(mask_tokens_out[:, -1, :]).unsqueeze(1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        large_embedding = self.output_upscaling_2(upscaled_embedding)
        large_hyper_in = self.output_hypernetworks_mlp_2(hyper_in)
        b, c, h, w = large_embedding.shape
        large_masks = (large_hyper_in @ large_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return masks, large_masks