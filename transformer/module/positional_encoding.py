import torch.nn as nn
import torch

"""
用于构造位置嵌入
位置编码是一种在模型中注入序列中各个元素位置信息的机制。
由于 Transformer 模型本身不包含循环或卷积，位置编码使模型能够考虑词语的顺序。

原文：
Since our model contains no recurrence and no convolution, in order for the model to make use of the
order of the sequence, we must inject some information about the relative or absolute position of the
tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the
bottoms of the encoder and decoder stacks. The positional encodings have the same dimension dmodel
as the embeddings, so that the two can be summed. There are many choices of positional encodings,
learned and fixed [9].
In this work, we use sine and cosine functions of different frequencies:
    PE(pos,2i) = sin(pos/10000^{2i/dmodel} )
    PE(pos,2i+1) = cos(pos/10000^{2i/dmodel} )
where pos is the position and i is the dimension. That is, each dimension of the positional encoding
corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 ·2π. We
chose this function because we hypothesized it would allow the model to easily learn to attend by
relative positions, since for any fixed offset k, PEpos+k can be represented as a linear function of PEpos.
We also experimented with using learned positional embeddings [9] instead, and found that the two
versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version
because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training

原文：
由于我们的模型不包含递归和卷积层，为了使模型能够利用序列的顺序信息，我们必须向模型注入关于序列中令牌相对或绝对位置的信息。
为此，我们在编码器和解码器堆栈的底部将“位置编码”添加到输入嵌入中。
位置编码的维度与嵌入的维度相同（d_model），因此两者可以相加。
位置编码有许多选择，包括学习得到的和固定的（参考文献 [9]）。
在这项工作中，我们使用了不同频率的正弦和余弦函数：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
其中，pos 是位置，i 是维度。也就是说，位置编码的每个维度都对应于一个正弦函数。
波长从 2π 到 10000 * 2π 形成几何级数。
我们选择这个函数是因为我们假设它可以让模型容易地通过相对位置进行关注，
因为对于任何固定的偏移量 k，PE(pos+k) 可以表示为 PE(pos) 的线性函数。
我们还尝试使用学习得到的位置嵌入（参考文献 [9]），
结果发现两种版本产生的结果几乎相同（见表 3，行 (E)）。
我们选择了正弦版本，因为它可能使模型能够推广到训练期间未遇到的更长的序列长度。
"""

class PositionalEncoding(nn.Module):
    """
    构造函数
    param d_model:  词向量的维度，即每个词汇的嵌入表示的大小
    param max_len:  一句话token最大个数
    """
    def __init__(self, d_model, max_len = 5000):
        # 父类先构造
        super(PositionalEncoding, self).__init__()

        # 生成维度递增序列 i，范围 0 - d_model
        seq_i = torch.arange(0, d_model, 2) # 注意步进为2，不然后续pe奇偶位置赋值会出现问题
        
        # 计算公式分母10000^(2i/d_model)
        # 形状 (d_model/2, )
        # div_term = torch.pow(10000, seq_i / d_model)
        div_term = torch.exp(seq_i * -(torch.log(torch.tensor(10000.0)) / d_model))   # 使用对数空间确保位置编码计算的数值稳定性，避免指数运算中的潜在数值问题

        # 生成位置序列 pos，范围 0 - max_len
        # 形状 (max_len, 1)
        pos = torch.arange(0, max_len).unsqueeze(1) # 注意扩展维度

        # 计算PE
        pe = torch.zeros(max_len, d_model)  # 形状 (max_len, d_model)
        """
        广播机制：
        pos(max_len, 1) / div_term(d_model/2, )
          div_term(d_model/2, ) -> div_term(1, d_model/2)
        pos(max_len, 1) / div_term(1, d_model/2) 
        ---> pos(max_len, d_model/2) / div_term(max_len, d_model/2)
        """
        pe[:,0::2] = torch.sin(pos / div_term)  # 偶数位置
        pe[:,1::2] = torch.cos(pos / div_term)  # 奇数位置
        pe.unsqueeze_(0)    # 形状 (1, max_len, d_model), 方便处理批量数据

        # 注册位置编码作为缓冲区，固定不可训练
        self.register_buffer('pe', pe)
    
    """
    前向传播函数（推理函数）
    param x: 词嵌入结果
    """
    def forward(self, x):
        # 词嵌入 + 位置编码
        # x(batch_size, len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x