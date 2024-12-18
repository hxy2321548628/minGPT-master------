"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from collections import namedtuple
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------
# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import math


# 定义各个激活函数
def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def gelu(x):
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )


# 生成数据
x = np.linspace(-5, 5, 500)  # 从-5到5，生成500个点
x_tensor = torch.tensor(x, dtype=torch.float32)

# 计算激活函数的值
relu_values = relu(x)
leaky_relu_values = leaky_relu(x)
gelu_values = gelu(x_tensor).numpy()  # torch tensor 转为 numpy array

# 绘制图像
plt.figure(figsize=(10, 6))
# plt.plot(x, relu_values, label="ReLU", color="blue", linestyle="-", linewidth=2)
plt.plot(
    x, leaky_relu_values, label="Leaky ReLU", color="green", linestyle="--", linewidth=2
)
plt.plot(x, gelu_values, label="GELU", color="red", linestyle="-.", linewidth=2)

# 设置标题和标签
plt.title("Activation Functions: ReLU, Leaky ReLU, GELU", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("f(x)", fontsize=14)
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)

# 添加图例
plt.legend()

# 显示图像
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------
# %%


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    """
    一个简单的多头掩蔽自注意力层，带有最终的投影。
    这段代码展示了如何手动实现自注意力层，虽然可以使用 `torch.nn.MultiheadAttention`，
    但这里提供了显式的实现来展示其原理。
    """

    def __init__(self, config):
        super().__init__()

        # 确保嵌入维度可以被头数整除，以便每个头的维度相同
        assert config.n_embd % config.n_head == 0

        # 线性变换，用于生成查询（Q）、键（K）、值（V）
        # 将输入的嵌入维度映射到 3 * n_embd，因为我们要输出三个部分：Q, K, V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # 输出投影层，作用是将最终的输出映射回嵌入维度
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # 丢弃层，防止过拟合
        self.attn_dropout = nn.Dropout(config.attn_pdrop)  # 自注意力的丢弃
        self.resid_dropout = nn.Dropout(config.resid_pdrop)  # 残差连接的丢弃

        # 因果掩蔽矩阵，确保每个位置只能看到它自己及其左边的部分
        # 使用下三角矩阵作为掩蔽，表示当前时间步只能注意到左边的内容
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        # 保存头数和嵌入维度
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        """
        前向传播函数
        输入：x -> 输入的张量，形状为 (B, T, C)，其中 B 是批量大小，T 是序列长度，C 是嵌入维度
        输出：y -> 自注意力层的输出，形状为 (B, T, C)
        """
        # 获取输入的批大小、序列长度和嵌入维度
        B, T, C = x.size()

        # 使用线性层生成查询、键和值的表示
        # 通过 `split(self.n_embd, dim=2)` 将 3 * n_embd 拆分为 Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 将键、查询和值的维度调整为 (B, nh, T, hs)
        # 其中 nh 是头数，hs 是每个头的维度 (C // n_head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # 计算注意力得分，采用缩放点积注意力
        # q @ k.transpose(-2, -1) 表示查询和键的点积
        # 用 sqrt(k.size(-1)) 进行缩放，避免数值过大
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 应用因果掩蔽，确保每个位置只能看到它自己以及之前的位置
        # 将掩蔽矩阵为 0 的部分填充为 -inf，防止计算 softmax 时参与计算
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # 对注意力得分进行 softmax，得到注意力权重
        att = F.softmax(att, dim=-1)

        # 应用注意力的丢弃（dropout）
        att = self.attn_dropout(att)

        # 将注意力权重应用到值上，计算最终的输出
        # 通过注意力权重与值相乘，得到每个位置的加权和
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # 重新排列多头输出，将它们拼接起来，最终的维度为 (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 将维度恢复到 (B, T, C)

        # 对输出进行投影（通过线性层），将其映射回原始嵌入维度，并应用残差连接的丢弃
        y = self.resid_dropout(self.c_proj(y))

        return y


class Block(nn.Module):
    """一个简单的 Transformer 块。"""

    def __init__(self, config):
        super().__init__()

        # 第一层规范化 (LayerNorm)
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # 自注意力层 (CausalSelfAttention)，用于计算输入的自注意力
        self.attn = CausalSelfAttention(config)

        # 第二层规范化 (LayerNorm)
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # 多层感知机 (MLP)，包括全连接层、激活函数和丢弃层
        self.mlp = nn.ModuleDict(
            dict(
                # 第一层全连接，输入为 n_embd，输出为 4 * n_embd
                c_fc=nn.Linear(config.n_embd, 4 * config.n_embd),
                # 第二层全连接，输入为 4 * n_embd，输出为 n_embd
                c_proj=nn.Linear(4 * config.n_embd, config.n_embd),
                # 激活函数，使用 NewGELU 激活
                act=NewGELU(),
                # 丢弃层，应用于 MLP 的输出
                dropout=nn.Dropout(config.resid_pdrop),
            )
        )

        # 定义 MLP 前向传播操作
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP 前向传播

    def forward(self, x):
        """
        前向传播函数。
        输入：x -> 输入的张量，形状为 (B, T, C)，其中 B 是批大小，T 是序列长度，C 是嵌入维度。
        输出：x -> 通过 Transformer 块的输出，形状与输入相同 (B, T, C)。
        """

        # 第一次残差连接：首先对输入进行 LayerNorm，然后通过自注意力层进行计算
        # self.attn(self.ln_1(x)) 对输入进行自注意力计算，结果与输入相加（残差连接）
        x = x + self.attn(self.ln_1(x))

        # 第二次残差连接：同样，首先进行 LayerNorm，然后通过 MLP 层进行计算
        # self.mlpf(self.ln_2(x)) 对输入进行 MLP 计算，结果与输入相加（残差连接）
        x = x + self.mlpf(self.ln_2(x))

        return x


# 创建一个简单的类 CN，用来存储配置参数
CN = namedtuple(
    "Config",
    [
        "model_type",
        "n_layer",
        "n_head",
        "n_embd",
        "vocab_size",
        "block_size",
        "embd_pdrop",
        "resid_pdrop",
        "attn_pdrop",
    ],
)


class GPT(nn.Module):
    """GPT 语言模型"""

    @staticmethod
    def get_default_config():
        """
        获取默认配置的方法。
        返回一个包含 GPT 模型配置的命名元组 CN。
        """
        C = CN(
            model_type="gpt",  # 模型类型，默认是 "gpt"
            n_layer=None,  # 模型的层数 (后续会由外部配置填充)
            n_head=None,  # 每层的头数 (后续会由外部配置填充)
            n_embd=None,  # 嵌入维度 (后续会由外部配置填充)
            vocab_size=None,  # 词汇表大小 (后续会由外部配置填充)
            block_size=None,  # 块大小，通常是序列长度 (后续会由外部配置填充)
            embd_pdrop=0.1,  # 嵌入层的丢弃概率
            resid_pdrop=0.1,  # 残差连接的丢弃概率
            attn_pdrop=0.1,  # 注意力层的丢弃概率
        )
        return C

    def __init__(self, config):
        """
        初始化 GPT 模型。
        参数：
            config: 包含配置参数的命名元组 CN。
        """
        super().__init__()

        # 确保配置中包含 vocab_size 和 block_size
        assert config.vocab_size is not None, "vocab_size must be specified."
        assert config.block_size is not None, "block_size must be specified."
        self.block_size = config.block_size  # 序列长度

        # 判断是通过 model_type 还是通过 n_layer, n_head, n_embd 来配置模型
        type_given = config.model_type is not None
        params_given = all(
            [
                config.n_layer is not None,
                config.n_head is not None,
                config.n_embd is not None,
            ]
        )

        # 通过异或操作保证这两个配置方式恰好有一个被提供
        assert (
            type_given ^ params_given
        ), "Exactly one of 'model_type' or ('n_layer', 'n_head', 'n_embd') must be specified."

        if type_given:
            # 根据 model_type 自动配置模型参数
            config.merge_from_dict(
                {
                    # GPT-1 配置
                    "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),  # 117M 参数
                    # GPT-2 配置
                    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M 参数
                    "gpt2-medium": dict(
                        n_layer=24, n_head=16, n_embd=1024
                    ),  # 350M 参数
                    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M 参数
                    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M 参数
                    # Gopher 配置
                    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
                    # 假设的一些小模型配置
                    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
                    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
                    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
                }[config.model_type]
            )

        # 创建 Transformer 模块
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # 词汇嵌入层
                wpe=nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入层
                drop=nn.Dropout(config.embd_pdrop),  # 嵌入层的丢弃层
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),  # Transformer 层列表
                ln_f=nn.LayerNorm(config.n_embd),  # 最后的 LayerNorm 层
            )
        )

        # 语言模型输出层
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化所有权重，并按 GPT-2 论文中的建议进行特殊的残差投影初始化
        self.apply(self._init_weights)  # 这里会遍历所有属性中的模型，然后进行初始化
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # 输出模型参数数量（不包括语言模型头中的参数）
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        """
        该方法用于初始化模型中的权重。根据模块类型（`nn.Linear`、`nn.Embedding`、`nn.LayerNorm`），
        采用不同的初始化策略。

        参数：
            module (nn.Module): 当前正在处理的模块（层）。
        """
        # 如果模块是一个线性层 (nn.Linear)
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 如果该层有偏置项，则将其初始化为0
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # 如果模块是一个嵌入层 (nn.Embedding)
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重，均值为0，标准差为0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # 如果模块是层归一化层 (nn.LayerNorm)
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0
            torch.nn.init.zeros_(module.bias)
            # 将权重项初始化为1
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        # 确保传入的模型类型是预定义的 GPT2 类型之一
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        # 从 Hugging Face 的 transformers 库中导入 GPT2LMHeadModel
        from transformers import GPT2LMHeadModel

        # 创建一个从头开始初始化的 minGPT 模型
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257  # OpenAI 的 GPT 模型词汇表大小
        config.block_size = 1024  # OpenAI 的 GPT 模型 block_size
        model = GPT(config)  # 使用配置初始化 GPT 模型
        sd = model.state_dict()  # 获取当前模型的状态字典（即模型的权重）

        # 初始化一个 Hugging Face 的预训练 GPT 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = (
            model_hf.state_dict()
        )  # 获取 Hugging Face 模型的状态字典（即预训练权重）

        # 对比两个模型的权重字典，并进行复制
        # 忽略以 "attn.masked_bias" 结尾的权重
        keys = [k for k in sd_hf if not k.endswith("attn.masked_bias")]

        # 对于某些层，我们需要进行权重转置
        transposed = [
            "attn.c_attn.weight",  # 自注意力层的权重
            "attn.c_proj.weight",  # 自注意力层的投影权重
            "mlp.c_fc.weight",  # MLP 层的全连接层权重
            "mlp.c_proj.weight",  # MLP 层的投影层权重
        ]

        # OpenAI 的模型使用了 Conv1D 层，我们的实现使用的是标准的 nn.Linear 层
        # 因此这些 Conv1D 层的权重需要转置后才能正确赋值
        assert len(keys) == len(sd)  # 确保两个权重字典的键数量一致

        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # 对于需要转置的权重，执行转置操作
                assert sd_hf[k].shape[::-1] == sd[k].shape  # 确保转置后的形状是匹配的
                with torch.no_grad():  # 不需要计算梯度
                    sd[k].copy_(sd_hf[k].t())  # 将转置后的权重拷贝到当前模型中
            else:
                # 对于不需要转置的普通权重，直接复制
                assert sd_hf[k].shape == sd[k].shape  # 确保形状一致
                with torch.no_grad():  # 不需要计算梯度
                    sd[k].copy_(sd_hf[k])  # 直接将权重拷贝

        return model  # 返回填充了预训练权重的模型

    def configure_optimizers(self, train_config):
        """
        这个函数将模型的参数分为两类：那些会进行权重衰减（regularization）的，和那些不会（比如偏置项、LayerNorm和Embedding层的权重）。
        然后返回一个 PyTorch 优化器对象（AdamW）。
        """

        # 定义两个集合：decay 和 no_decay，分别用于存放会和不会进行权重衰减的参数
        decay = set()
        no_decay = set()

        # 设置哪些层的权重需要进行衰减
        whitelist_weight_modules = (torch.nn.Linear,)
        # 设置哪些层的权重不需要进行衰减
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # 遍历模型中的每个子模块和参数
        for mn, m in self.named_modules():  # named_modules 返回的是 (模块名, 模块) 对
            for (
                pn,
                p,
            ) in m.named_parameters():  # named_parameters 返回的是 (参数名, 参数) 对
                fpn = "%s.%s" % (mn, pn) if mn else pn  # 拼接出完整的参数名称

                # 处理偏置项（不进行权重衰减）
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                # 处理 Linear 层的权重（会进行权重衰减）
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                # 处理 LayerNorm 和 Embedding 层的权重（不进行权重衰减）
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # 确保所有参数都已正确分类
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay  # 检查是否有参数同时出现在 decay 和 no_decay 中
        union_params = decay | no_decay  # 获取所有涉及到的参数的并集

        # 如果某个参数既出现在 decay 集合又出现在 no_decay 集合中，则报错
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        # 确保所有参数都已经被分配到 decay 或 no_decay 中
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # 创建 PyTorch 的优化器对象
        optim_groups = [
            {
                "params": [
                    param_dict[pn] for pn in sorted(list(decay))
                ],  # 所有需要衰减的参数
                "weight_decay": train_config.weight_decay,  # 设置权重衰减系数
            },
            {
                "params": [
                    param_dict[pn] for pn in sorted(list(no_decay))
                ],  # 所有不需要衰减的参数
                "weight_decay": 0.0,  # 不进行权重衰减
            },
        ]

        # 使用 AdamW 优化器，设置学习率、动量等超参数
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )

        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device  # 获取输入张量 idx 所在的设备 (CPU or GPU)
        b, t = idx.size()  # b 是批次大小，t 是序列长度
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"  # 确保序列长度不超过 block_size

        # 创建位置编码，shape (1, t)，表示每个 token 在序列中的位置
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        # 计算 token 和位置的嵌入
        tok_emb = self.transformer.wte(idx)  # 获取 token 的嵌入，shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # 获取位置嵌入，shape (1, t, n_embd)

        # 将 token 和位置嵌入相加，并进行 Dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # 将输入传递通过多个 transformer block
        for block in self.transformer.h:
            x = block(x)

        # 最后通过一个 LayerNorm
        x = self.transformer.ln_f(x)

        # 通过语言模型头（lm_head）生成 logits
        logits = self.lm_head(x)

        # 如果提供了 targets，则计算交叉熵损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None
    ):
        """
        给定一个条件序列 `idx`（形状为 (b, t) 的 LongTensor），并完成生成 `max_new_tokens` 次，
        每次将预测结果馈送回模型。你通常希望在模型的 `eval()` 模式下运行此函数。
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，则需要截断至 `block_size`
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )

            # 前向传播，获取序列中当前位置的 logits
            logits, _ = self(idx_cond)

            # 获取最后一个 token 的 logits，并按所需的温度进行缩放
            logits = logits[:, -1, :] / temperature

            # 可选：对 logits 进行 top_k 剪枝，保留前 k 个最可能的选项
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float(
                    "Inf"
                )  # 将不在 top_k 之内的 logits 置为负无穷

            # 使用 softmax 将 logits 转换为概率分布
            probs = F.softmax(logits, dim=-1)

            # 根据 `do_sample` 决定是采样还是选择最可能的元素
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)  # 从分布中进行采样
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)  # 选择概率最大的索引

            # 将生成的 token 索引追加到当前序列，并继续生成
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
