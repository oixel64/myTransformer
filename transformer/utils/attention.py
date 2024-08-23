import torch.nn as nn
import torch
import torch.nn.functional as F

"""
注意力计算

原文：
Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of
queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the
query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the
values.

In practice, we compute the attention function on a set of queries simultaneously, packed together
into a matrix Q. The keys and values are also packed together into matrices K and V . We compute
the matrix of outputs as:
    Attention(Q,K,V ) = softmax(QK^T/√dk)V

The two most commonly used attention functions are additive attention [2], and dot-product (multi-
plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor
of 1/√dk. Additive attention computes the compatibility function using a feed-forward network with
a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is
much faster and more space-efficient in practice, since it can be implemented using highly optimized
matrix multiplication code.

While for small values of dk the two mechanisms perform similarly, additive attention outperforms
dot product attention without scaling for larger values of dk [3]. We suspect that for large values of
dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has
extremely small gradients 4. To counteract this effect, we scale the dot products by 1/√dk.

译文：
缩放点积注意力

我们称我们特定的注意力机制为“缩放点积注意力”。
输入包括查询（queries）和键（keys），它们的维度为 d_k，而值（values）的维度为 d_v。
我们计算查询与所有键的点积，将每个点积除以 sqrt{d_k}，然后应用 softmax 函数来获得值的权重。

在实际操作中，我们同时对一组查询进行计算，将其打包成一个矩阵 Q。
键和值也被打包成矩阵 K 和 V。我们计算输出矩阵的方法为：
    Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V

最常用的两种注意力机制是加性注意力（additive attention）和点积（乘法）注意力（dot-product attention）。
点积注意力与我们的算法相同，只是没有缩放因子 1\sqrt{d_k}。
加性注意力使用一个具有单个隐藏层的前馈网络来计算兼容性函数。
虽然这两种机制在理论复杂性上类似，但点积注意力在实际中由于可以使用高度优化的矩阵乘法代码，计算速度更快、空间效率更高。

对于较小的 d_k 值，这两种机制表现类似，
但在 d_k 较大时，加性注意力在没有缩放的情况下表现优于点积注意力。
我们怀疑，对于较大的 d_k 值，点积的值会增大，推使 softmax 函数进入极小梯度的区域。
为了对抗这种效果，我们将点积缩放为 1\sqrt{d_k}。
"""

# Attention
def attention(Q, K, V, mask=None):
    # 计算注意力得分
    d_k = torch.tensor(K.shape[-1])
    scores = Q @ K.transpose(-2, -1) / torch.sqrt(d_k)

    # 遮蔽
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9) # 注意是一个负的极大数  

    # softmax
    p = F.softmax(scores, dim=-1)   # 注意是沿最后一个维度做softmax

    return p @ V