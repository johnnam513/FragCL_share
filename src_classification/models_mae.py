# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm_vision_transformer import Block
import torch.nn.functional as F
#from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        #self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        #num_patches = self.patch_embed.num_patches

        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        #self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim*in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        #decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        #self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        #w = self.patch_embed.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        #torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        
        N, L, D = x.shape  # batch, length, dim
        x1 = x[:,:L//2,:]
        x2 = x[:,L//2:,:]
        
        L = L//2
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x1_masked = torch.gather(x1, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x2_masked = torch.gather(x2, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x1.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        x_masked = torch.cat([x1_masked, x2_masked], dim=1)
        
        ids_restore2 = ids_restore * 2 + 1
        ids_restore = ids_restore * 2
        ids_restore = torch.cat([ids_restore, ids_restore2], dim=1)
        mask = torch.cat([mask, mask], dim=1)
        return x_masked, mask, ids_restore

    def random_masking_before(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio):
        # embed patches
        #x = self.patch_embed(x)

        # add pos embed w/o cls token
        #x = x + self.pos_embed[:, 1:, :]

        x = self.norm(x)
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # append cls token
        #cls_token = self.cls_token + self.pos_embed[:, :1, :]
        #cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        #x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        #mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)

        #x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token

        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = x_
        #x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        #x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        #x = x[:, 1:, :]

        return x

    def forward_loss_before(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        #target = self.patchify(imgs)
        target = imgs
        #if self.norm_pix_loss:
        #mean = target.mean(dim=-1, keepdim=True)
        #var = target.var(dim=-1, keepdim=True)
        #target = (target - mean) / (var + 1.e-6)**.5
        N,L,W = pred.shape
        pred1 = pred[:,:L//2,:]
        pred2 = pred[:,L//2:,:]
        target1 = target[:,:L//2,:]
        target2 = target[:,L//2:,:]

        N,L,W = mask.shape
        mask = mask[:,:L//2,:]


        target = F.normalize(target, dim=2)
        pred = F.normalize(pred, dim=2)
        print(mask.shape)
        target = target[mask==1]
        pred = pred[mask==1]
        print(target.shape)
        print(pred.shape)
        raise dd

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        #target = self.patchify(imgs)
        target = imgs
        #if self.norm_pix_loss:
        #mean = target.mean(dim=-1, keepdim=True)
        #var = target.var(dim=-1, keepdim=True)
        #target = (target - mean) / (var + 1.e-6)**.5
        N,L,W = pred.shape
        pred1 = pred[:,:L//2,:]
        pred2 = pred[:,L//2:,:]
        target1 = target[:,:L//2,:]
        target2 = target[:,L//2:,:]

        N,L = mask.shape
        mask = mask[:,:L//2]


        #target = F.normalize(target, dim=2)
        #pred = F.normalize(pred, dim=2)
        
        target1 = target1[mask==1]
        pred1 = pred1[mask==1]
        
        
        target2 = target2[mask==1]
        pred2 = pred2[mask==1]

        T = 0.1
        batch, _ = target1.size()
        x1_abs = target1.norm(dim=1)
        x2_abs = pred1.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', target1, pred1) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)


        sim_matrix = torch.exp(sim_matrix / T)
        
        answer1 = torch.sum(torch.eq(sim_matrix.argmax(dim=1),torch.Tensor(range(batch)).cuda()))
        

        pos_sim = sim_matrix[range(batch), range(batch)]
        loss1 = (pos_sim) / (sim_matrix.sum(dim=1) - pos_sim)
        
        loss1 = - torch.log(loss1).mean()

        batch, _ = target2.size()
        x1_abs = target2.norm(dim=1)
        x2_abs = pred2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', target2, pred2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)


        sim_matrix = torch.exp(sim_matrix / T)
        answer2 = torch.sum(torch.eq(sim_matrix.argmax(dim=1),torch.Tensor(range(batch)).cuda()))

        pos_sim = sim_matrix[range(batch), range(batch)]
        loss2 = (pos_sim) / (sim_matrix.sum(dim=1) - pos_sim)
        
        loss2 = - torch.log(loss2).mean()

        #loss = (pred - target) ** 2
        #loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        #loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        loss = loss1 + loss2
        return loss, answer1, answer2

    def forward(self, imgs, mask_ratio=0.25):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss, answer1, answer2 = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, answer1, answer2


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
