"""
Vision Transformer 模型实现
该文件实现了用于超表面设计的Vision Transformer模型。
包括DropPath、多层感知机(MLP)、多头自注意力机制、Transformer块等核心组件。
适配Jones矩阵数据的特殊输入格式，支持绝对位置编码和相对位置偏置。
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """
    DropPath（随机深度）
    
    在残差块的主路径上按样本随机丢弃路径，用于正则化和提高模型泛化能力。
    在训练时随机将某些样本的残差路径置零，测试时不进行丢弃。
    """
    
    def __init__(self, drop_prob=None):
        """
        初始化DropPath
        
        参数:
            drop_prob: 路径丢弃概率
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            应用DropPath后的张量
        """
        # 不丢弃路径的情况：概率为0或处于评估模式
        if self.drop_prob == 0. or not self.training:
            return x
        
        # 计算保留概率
        keep_prob = 1 - self.drop_prob
        # 创建随机mask，shape与batch维度相同，其他维度为1（用于广播）
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化：大于等于1的为1，小于1的为0
        # 按keep_prob缩放并应用mask
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """
    多层感知机(MLP)模块
    
    用于Transformer块中的前馈神经网络部分。
    结构：Linear -> Activation -> Dropout -> Linear -> Dropout
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        初始化MLP
        
        参数:
            in_features: 输入特征维度
            hidden_features: 隐藏层特征维度，默认与输入相同
            out_features: 输出特征维度，默认与输入相同
            act_layer: 激活函数，默认为GELU
            drop: Dropout概率
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 第一个线性层：扩展维度
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数
        self.act = act_layer()
        # 第二个线性层：恢复维度
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout层
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，shape为[batch_size, seq_len, in_features]
            
        返回:
            输出张量，shape为[batch_size, seq_len, out_features]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    多头自注意力机制
    
    实现标准的多头自注意力，支持可选的相对位置偏置。
    使用QKV分解和缩放点积注意力。
    """
    
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        """
        初始化注意力模块
        
        参数:
            dim: 输入特征维度
            num_heads: 注意力头数
            qkv_bias: 是否对QKV使用偏置（仅对Q和V使用，K不使用）
            qk_scale: 注意力缩放因子，默认为head_dim的-0.5次方
            attn_drop: 注意力dropout概率
            proj_drop: 输出投影dropout概率
            window_size: 窗口大小，用于相对位置编码
            attn_head_dim: 每个注意力头的维度，默认为dim//num_heads
        """
        super().__init__()
        self.num_heads = num_heads
        # 计算每个注意力头的维度
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # 缩放因子：1/sqrt(head_dim)
        self.scale = qk_scale or head_dim ** -0.5
        
        # QKV投影层（线性变换：Y_{n×o} = X_{n×i} @ W_{i×o} + b）
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        
        # 可选的QKV偏置（仅对Q和V使用偏置，K不使用以保持对称性）
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        # 相对位置偏置（可选）
        if window_size:
            self.window_size = window_size
            # 计算相对位置距离的数量
            # 包括：token到token、cls到token、token到cls、cls到cls
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            # 相对位置偏置表
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # shape: [2*Wh-1 * 2*Ww-1 + 3, num_heads]

            # 计算相对位置索引
            # 生成窗口内每个位置的坐标
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # shape: [2, Wh, Ww]
            coords_flatten = torch.flatten(coords, 1)  # shape: [2, Wh*Ww]
            # 计算相对坐标
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # shape: [2, Wh*Ww, Wh*Ww]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # shape: [Wh*Ww, Wh*Ww, 2]
            # 偏移使坐标从0开始
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            # 转换为一维索引
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            # 创建相对位置索引（包含cls token）
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # token到token
            relative_position_index[0, 0:] = self.num_relative_distance - 3  # cls到token
            relative_position_index[0:, 0] = self.num_relative_distance - 2  # token到cls
            relative_position_index[0, 0] = self.num_relative_distance - 1   # cls到cls

            # 注册为buffer（不参与梯度更新）
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        # Dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影层
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        """
        前向传播
        
        计算多头自注意力，支持相对位置偏置。
        
        参数:
            x: 输入张量，shape为[B, N, C]，B为batch size，N为序列长度，C为特征维度
            rel_pos_bias: 可选的相对位置偏置，shape为[num_heads, N, N]
            
        返回:
            输出张量，shape为[B, N, C]
        """
        B, N, C = x.shape
        
        # 构建QKV偏置（仅对Q和V使用偏置）
        qkv_bias = None
        if self.q_bias is not None:
            # K的偏置为0（保持对称性）
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        
        # 计算QKV：线性变换 + 偏置
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # 重塑并分离QKV：[B, N, 3*all_head_dim] -> [3, B, num_heads, N, head_dim]
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个的shape: [B, num_heads, N, head_dim]
        
        # 缩放Q（注意力缩放）
        q = q * self.scale
        # 计算注意力分数：Q @ K^T
        attn = (q @ k.transpose(-2, -1))  # shape: [B, num_heads, N, N]

        # 添加相对位置偏置（如果有）
        if self.relative_position_bias_table is not None:
            # 根据相对位置索引查表获取偏置
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # shape: [N, N, num_heads]
            # 调整维度顺序：[num_heads, N, N]
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            # 添加到注意力分数（增加batch维度用于广播）
            attn = attn + relative_position_bias.unsqueeze(0)

        # 添加外部提供的相对位置偏置（如果有）
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        # 应用softmax得到注意力权重
        attn = attn.softmax(dim=-1)
        # Dropout
        attn = self.attn_drop(attn)

        # 注意力加权求和：Attention @ V
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)  # shape: [B, N, all_head_dim]
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer块
    
    标准的Transformer编码器块，包含多头自注意力和MLP，使用残差连接和LayerNorm。
    可选的LayerScale用于稳定训练。
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        """
        初始化Transformer块
        
        参数:
            dim: 输入特征维度
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层维度与输入维度的比率
            qkv_bias: 是否对QKV使用偏置
            qk_scale: 注意力缩放因子
            drop: Dropout概率
            attn_drop: 注意力dropout概率
            drop_path: DropPath概率
            init_values: LayerScale初始化值，None表示不使用
            act_layer: 激活函数
            norm_layer: 归一化层
            window_size: 窗口大小（用于相对位置编码）
            attn_head_dim: 每个注意力头的维度
        """
        super().__init__()
        # 第一个LayerNorm（注意力之前）
        self.norm1 = norm_layer(dim)
        # 多头自注意力
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # DropPath（随机深度）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二个LayerNorm（MLP之前）
        self.norm2 = norm_layer(dim)
        # MLP（前馈神经网络）
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # LayerScale：可学习的缩放因子，用于稳定深层网络的训练
        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        """
        前向传播
        
        参数:
            x: 输入张量，shape为[B, N, C]
            rel_pos_bias: 可选的相对位置偏置
            
        返回:
            输出张量，shape为[B, N, C]
        """
        # 残差连接 + 注意力
        if self.gamma_1 is None:
            # 不使用LayerScale
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            # 残差连接 + MLP
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # 使用LayerScale
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    Patch嵌入层
    
    将输入数据（Jones矩阵）划分为patches并嵌入到高维空间。
    对于Jones矩阵数据，通常使用1x1的patch size，相当于逐点嵌入。
    
    注意：未来可能需要处理更复杂的Jones矩阵维度，可能会使用更大的patch size。
    """
    
    def __init__(self, x_size=20, y_size=6, patch_size=1, in_chans=1, embed_dim=768):
        """
        初始化Patch嵌入层
        
        参数:
            x_size: 输入数据的X维度大小（波长点数）
            y_size: 输入数据的Y维度大小（Jones矩阵展平后的维度）
            patch_size: Patch大小
            in_chans: 输入通道数
            embed_dim: 嵌入维度
        """
        super().__init__()
        img_size = (x_size, y_size)
        patch_size = (patch_size, patch_size)
        # 计算patch数量
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # Patch网格的形状
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # 使用卷积实现patch嵌入（kernel_size和stride都等于patch_size）
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，shape为[B, C, H, W]
            
        返回:
            嵌入后的张量，shape为[B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        # 检查输入尺寸是否匹配
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # Patch嵌入流程：
        # 1. proj: 卷积操作 [B, C, H, W] -> [B, embed_dim, H', W']
        # 2. flatten: 展平空间维度 [B, embed_dim, H', W'] -> [B, embed_dim, H'*W']
        # 3. transpose: 交换维度 [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):
    """
    相对位置偏置模块
    
    用于所有Transformer层共享的相对位置偏置。
    计算窗口内每对位置之间的相对位置偏置。
    """
    
    def __init__(self, window_size, num_heads):
        """
        初始化相对位置偏置
        
        参数:
            window_size: 窗口大小，tuple格式(height, width)
            num_heads: 注意力头数
        """
        super().__init__()
        self.window_size = window_size
        # 计算相对位置距离的数量（包括cls token）
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 相对位置偏置表（可学习参数）
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # shape: [2*Wh-1 * 2*Ww-1 + 3, num_heads]
        
        # 计算相对位置索引（与Attention类中的计算相同）
        # 生成窗口内每个位置的坐标
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # shape: [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # shape: [2, Wh*Ww]
        # 计算相对坐标
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # shape: [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # shape: [Wh*Ww, Wh*Ww, 2]
        # 偏移使坐标从0开始
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        # 转换为一维索引
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        # 创建相对位置索引（包含cls token）
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # token到token
        relative_position_index[0, 0:] = self.num_relative_distance - 3  # cls到token
        relative_position_index[0:, 0] = self.num_relative_distance - 2  # token到cls
        relative_position_index[0, 0] = self.num_relative_distance - 1   # cls到cls

        # 注册为buffer（不参与梯度更新）
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        """
        前向传播
        
        返回:
            相对位置偏置，shape为[num_heads, N, N]，N为窗口内的位置数（包括cls token）
        """
        # 根据相对位置索引从偏置表中查找对应的偏置值
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # shape: [N, N, num_heads]
        # 调整维度顺序以匹配注意力计算
        return relative_position_bias.permute(2, 0, 1).contiguous()  # shape: [num_heads, N, N]


class VisionTransformer(nn.Module):
    """
    Vision Transformer 主模型
    
    适配超表面Jones矩阵数据的Vision Transformer模型。
    用于从Jones矩阵预测结构参数或进行预训练。
    
    主要组件：
    - Patch嵌入：将Jones矩阵数据转换为token序列
    - 位置编码：绝对位置编码或相对位置偏置
    - Transformer编码器：多层Transformer块
    - 分类头：输出结构参数预测
    """
    
    def __init__(self, x_size=20, y_size=6, patch_size=1, in_chans=1, num_para=6, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001):
        """
        初始化Vision Transformer模型
        
        参数:
            x_size: Jones矩阵X维度大小（波长点数）
            y_size: Jones矩阵Y维度大小（展平后的维度）
            patch_size: Patch大小
            in_chans: 输入通道数
            num_para: 输出参数数量（结构参数），0表示不使用分类头（预训练模式）
            embed_dim: 嵌入维度
            depth: Transformer层数
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层维度比率
            qkv_bias: 是否对QKV使用偏置
            qk_scale: 注意力缩放因子
            drop_rate: Dropout比率
            attn_drop_rate: 注意力Dropout比率
            drop_path_rate: DropPath比率
            norm_layer: 归一化层
            init_values: LayerScale初始化值
            use_abs_pos_emb: 是否使用绝对位置编码
            use_rel_pos_bias: 是否使用相对位置偏置
            use_shared_rel_pos_bias: 是否使用共享的相对位置偏置
            use_mean_pooling: 是否使用平均池化（否则使用cls token）
            init_scale: 分类头权重初始化缩放因子
        """
        super().__init__()
        self.num_para = num_para  # 输出参数数量
        self.num_features = self.embed_dim = embed_dim  # 特征维度
        self.patch_size = patch_size
        self.in_chans = in_chans

        # Patch嵌入层
        self.patch_embed = PatchEmbed(
            x_size=x_size, y_size=y_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token：用于分类的特殊token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 绝对位置编码（可选）
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 共享的相对位置偏置（可选）
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        # DropPath比率的随机深度衰减规则（从0线性增长到drop_path_rate）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        
        # 构建Transformer块列表
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])
        
        # 归一化层和分类头
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_para) if num_para > 0 else nn.Identity()

        # 权重初始化
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        if num_para > 0:
            nn.init.trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        # 缩放分类头权重（用于更稳定的训练）
        if num_para > 0:
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        """
        修正初始化权重
        
        对每层的注意力投影和MLP输出层权重进行重新缩放。
        使用层级相关的缩放因子，有助于训练深层网络。
        """
        def rescale(param, layer_id):
            """根据层级ID重新缩放参数"""
            param.div_(math.sqrt(2.0 * layer_id))

        # 对每个Transformer块的投影层进行缩放
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @staticmethod
    def _init_weights(m):
        """
        权重初始化函数
        
        为不同类型的层应用适当的初始化策略。
        
        参数:
            m: 神经网络模块
        """
        if isinstance(m, nn.Linear):
            # 线性层：截断正态分布初始化
            nn.init.trunc_normal_(m.weight, std=.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            # LayerNorm：偏置初始化为0，权重初始化为1
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            # 卷积层：Kaiming初始化
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_num_layers(self):
        """获取Transformer层数"""
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        """返回不应用权重衰减的参数名称"""
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        """获取分类头"""
        return self.head

    def reset_para(self, num_para, global_pool=''):
        """
        重置输出参数数量
        
        用于改变模型输出的参数数量（例如从预训练切换到微调）。
        
        参数:
            num_para: 新的输出参数数量
            global_pool: 全局池化方式（预留参数）
        """
        self.num_para = num_para
        self.head = nn.Linear(self.embed_dim, num_para) if num_para > 0 else nn.Identity()

    def forward_features(self, x):
        """
        提取特征
        
        通过Transformer编码器提取特征向量。
        
        参数:
            x: 输入张量，shape为[B, C, H, W]
            
        返回:
            特征向量，shape为[B, embed_dim]
        """
        # Patch嵌入：[B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置编码
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # 计算共享的相对位置偏置（如果使用）
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        
        # 通过所有Transformer块
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        # 归一化
        x = self.norm(x)
        
        # 提取特征向量
        if self.fc_norm is not None:
            # 使用平均池化：对所有patch token（排除cls token）求平均
            t = x[:, 1:, :]
            return self.fc_norm(t.mean(1))
        else:
            # 使用CLS token
            return x[:, 0]

    def forward(self, x):
        """
        前向传播
        
        完整的前向传播过程：提取特征 -> 分类头预测参数。
        
        参数:
            x: 输入张量，shape为[B, C, H, W]
            
        返回:
            预测的结构参数，shape为[B, num_para]
        """
        # 提取特征
        x = self.forward_features(x)
        # 通过分类头预测参数：[B, embed_dim] -> [B, num_para]
        x = self.head(x)
        return x


def build_vit(config):
    """
    构建Vision Transformer模型
    
    根据配置对象创建Vision Transformer模型实例。
    
    参数:
        config: 配置对象，包含模型的所有超参数
        
    返回:
        配置好的VisionTransformer模型
    """
    model = VisionTransformer(
        x_size=config.DATA.SIZE_X,
        y_size=config.DATA.SIZE_Y,
        patch_size=config.MODEL.VIT.PATCH_SIZE,
        in_chans=config.MODEL.VIT.IN_CHANS,
        num_para=config.MODEL.NUM_PARA,
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

    return model
