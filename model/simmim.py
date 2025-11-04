"""
SimMIM 预训练模型
该文件实现了用于Masked Image Modeling (MIM)预训练的模型。
SimMIM通过掩码部分输入并重建来进行自监督学习，适用于Jones矩阵数据的预训练。

主要组件：
- VisionTransformerForSimMIM：带掩码token的Vision Transformer编码器
- SimMIM：完整的掩码图像建模框架，包括编码器和解码器
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import VisionTransformer


class VisionTransformerForSimMIM(VisionTransformer):
    """
    用于SimMIM的Vision Transformer
    
    继承自标准Vision Transformer，添加了掩码token用于掩码输入的重建。
    在预训练阶段使用，不需要分类头（num_para=0）。
    """
    
    def __init__(self, **kwargs):
        """
        初始化SimMIM版本的Vision Transformer
        
        参数:
            **kwargs: Vision Transformer的所有参数
        """
        super().__init__(**kwargs)
        # 确保预训练模式（不使用分类头）
        assert self.num_para == 0

        # 掩码token：用于替换被掩码的patch
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # 初始化掩码token
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, mask):
        """
        前向传播
        
        将被掩码的patch替换为掩码token，然后通过Transformer编码器。
        
        参数:
            x: 输入张量，shape为[B, C, H, W]
            mask: 掩码张量，shape为[B, 1, H, W]，1表示被掩码，0表示保留
            
        返回:
            编码后的特征图，shape为[B, C, H, W]
        """
        _, _, H, W = x.shape
        # Patch嵌入：[B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)

        # 确保提供了掩码
        assert mask is not None
        B, L, _ = x.shape  # B: batch size, L: 序列长度（num_patches）

        # 应用掩码：将被掩码的patch替换为掩码token
        mask_token = self.mask_token.expand(B, L, -1)  # 扩展到batch size
        # w: 掩码权重，1表示使用掩码token，0表示使用原始patch
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        # 混合原始patch和掩码token
        x = x * (1 - w) + mask_token * w
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过Transformer编码器
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        # 移除CLS token，只保留patch tokens
        x = x[:, 1:]
        # 重塑为特征图：[B, L, C] -> [B, C, H, W]
        B, L, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class SimMIM(nn.Module):
    """
    SimMIM 模型
    
    完整的Masked Image Modeling框架，包括：
    - 编码器：VisionTransformerForSimMIM
    - 解码器：简单的1x1卷积
    - 损失函数：L1损失，支持不同的计算方式
    """
    
    def __init__(self, encoder, loss_type, is_recon):
        """
        初始化SimMIM模型
        
        参数:
            encoder: VisionTransformerForSimMIM编码器
            loss_type: 损失计算类型
                0: 计算整个Jones矩阵的损失
                1: 仅计算被掩码部分的损失
                2: 仅计算未被掩码部分的损失
            is_recon: 是否为重建模式（返回重建结果）
        """
        super().__init__()
        self.encoder = encoder
        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
        self.loss_type = loss_type
        self.is_recon = is_recon
        # 解码器：1x1卷积，将encoder输出映射回输入空间
        self.decoder = nn.Conv2d(in_channels=self.encoder.num_features, out_channels=self.in_chans, kernel_size=1)

    def forward(self, x, mask):
        """
        前向传播
        
        参数:
            x: 输入Jones矩阵，shape为[B, C, H, W]
            mask: 掩码，shape为[B, 1, H, W]
            
        返回:
            如果is_recon=True: (损失值, 重建结果)
            如果is_recon=False: 损失值
        """
        # 编码
        z = self.encoder(x, mask)
        # 解码
        x_rec = self.decoder(z)
        
        # 将掩码重复插值以匹配patch size
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).contiguous()
        
        # 计算损失
        if self.loss_type == 0:
            # 计算整个Jones矩阵的平均L1损失
            loss = F.l1_loss(x, x_rec, reduction='mean')
        elif self.loss_type == 1:
            # 仅计算被掩码部分的损失
            loss_recon = F.l1_loss(x, x_rec, reduction='none')
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        elif self.loss_type == 2:
            # 仅计算未被掩码部分的损失
            loss_recon = F.l1_loss(x, x_rec, reduction='none')
            mask = 1 - mask  # 反转掩码
            loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        else:
            raise ValueError("Loss Type must be 0, 1, 2!")
        
        # 返回结果
        if self.is_recon:
            return loss, x_rec
        else:
            return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        """返回不应用权重衰减的参数名称"""
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """返回不应用权重衰减的参数关键词"""
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(config):
    """
    构建SimMIM模型
    
    根据配置对象创建SimMIM预训练模型。
    
    参数:
        config: 配置对象
        
    返回:
        配置好的SimMIM模型
    """
    # 创建编码器
    encoder = VisionTransformerForSimMIM(
        x_size=config.DATA.SIZE_X,
        y_size=config.DATA.SIZE_Y,
        patch_size=config.MODEL.VIT.PATCH_SIZE,
        in_chans=config.MODEL.VIT.IN_CHANS,
        num_para=0,  # 预训练模式，不使用分类头
        embed_dim=config.MODEL.VIT.EMBED_DIM,
        depth=config.MODEL.VIT.DEPTH,
        num_heads=config.MODEL.VIT.NUM_HEADS,
        mlp_ratio=config.MODEL.VIT.MLP_RATIO,
        qkv_bias=config.MODEL.VIT.QKV_BIAS,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=config.MODEL.VIT.INIT_VALUES,
        use_abs_pos_emb=config.MODEL.VIT.USE_APE,
        use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
        use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
        use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)

    # 创建完整的SimMIM模型
    model = SimMIM(encoder, config.MODEL.LOSS_TYPE, config.RECON_MODE)

    return model
