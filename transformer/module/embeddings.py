import torch.nn as nn
import torch

"""
用于创建词嵌入
词嵌入是一种将词汇表中的单词（整数索引）转换为固定大小的高维向量的技术，
这些向量捕获了单词的语义信息。

原文：
Similarly to other sequence transduction models, we use learned embeddings to convert the input
tokens and output tokens to vectors of dimension dmodel. We also use the usual learned linear transfor-
mation and softmax function to convert the decoder output to predicted next-token probabilities. In
our model, we share the same weight matrix between the two embedding layers and the pre-softmax
linear transformation, similar to [30]. In the embedding layers, we multiply those weights by √dmodel.

译文：
与其他序列转换模型类似，我们使用学习到的嵌入将输入和输出的令牌转换为维度为 d_model 的向量。
我们还使用常见的线性变换和 softmax 函数将解码器的输出转换为预测的下一个令牌的概率。
在我们的模型中，我们在两个嵌入层和 softmax 之前的线性变换之间共享同一个权重矩阵，类似于 [30]。
在嵌入层中，我们将这些权重乘以 √d_model
"""

# Input Embedding
class Embeddings(nn.Module):
    """
    构造函数
    param vocab:    词汇表大小，即可以嵌入的不同词汇数量
    param d_model:  词向量的维度，即每个词汇的嵌入表示的大小
    """
    def __init__(self, vocab, d_model):
        # 调用父类的构造函数
        super(Embeddings, self).__init__()
        
        # 创建一个嵌入层look-up table，LUT
        # 嵌入层是一个矩阵，每行是一个词都向量表示
        self.lut = nn.Embedding(vocab, d_model)
        
        # 维度
        self.d_model = d_model  
    
    """
    前向传播函数（推理函数）
    param x: 整数索引的张量，代表单词
    """
    def forward(self, x):
        # 查找嵌入查找表
        lut_x =  self.lut(x)
        # 放大嵌入，保持合适梯度（标准做法）在初始化时保持嵌入向量的方差
        return lut_x * torch.sqrt(self.d_model)