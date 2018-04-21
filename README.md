# DataProcessing
#此代码用于模型训练前的数据预处理的典型过程，
包括：
1.建立Word2id 和 id2word 的字典
2.batch切分
3.最大长度截断

##code方面：
这是通过RNN+Attention进行的语义相似度的模型，其精妙之处在于：
1.通过BILSTM进行上下文语义的完整理解
2.通过4种Attension方式选择最佳的方式进行Query与Query之间的联系与Attention

