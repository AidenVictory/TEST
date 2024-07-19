---
title: NLP
updated: 2024-07-11 13:12:32Z
created: 2024-06-06 02:32:20Z
---

## 缩放点积注意力机制

\--Scaled Dot-Product Attention--

- token, 文本基本单位
- embedding, 将token映射到连续向量空间的结果
    - 解决矩阵稀疏，可降低维度
- encoding, 将输入数据转换为低维度、紧凑表示

### Attention

- Query,查询向量，翻译任务中的目标语言单词
- Key，键向量， 翻译任务中的源语言中的单词
- Value，值向量，根据查询向量和键向量的匹配程度来加权求和的信息，如翻译任务中源语言单词的嵌入向量

### 词向量

- 离散表示(one-hot representation),特征空间大但很多应用任务线性可分
- 分布式表示(distribution representation),定长连续稠密向量,词向量存在相似关系包含信息多
- 词向量生成，词向量查询表(Embedding Lookup),

> one-hot Encoding V &lt;dot&gt; Embedding lookup =

### 语义信息学习模型

word2vec 算法通过上下文学习语义信息

- CBOW(Continutous Bag-of-Words)

> 更快，使用上下文average的方式进行训练，每个训练step见到更多样本

- Skip-gram

> 不会回避生僻字，CBOW中生僻字会被其他非生僻字权重冲淡

## 李宏毅 自注意力机制

- 万物皆向量

> 一个句子就是长短不一的向量组，vector set  
> 音频，25ms，为frame，相当于向量表示  
> 图，每个节点视为一个向量，如分子

- 输出
    
    - Each vector has a label. Sequence labeling
        
        > 数值则为regression，class则为classification  
        > POS tagging,词性标注；HW2，音标语音辨识；
        
    - The whole sequence has a label
        
        > Sentiment analysis, positive or negative; HW4,audio-speaker; 分子，亲水性/疏水性；
        
    - Model decides the number of labels itself. Seq2Seq
        
        > translation(HW5)
        
- Sequence Labeling
    
    - Fully connected, window 考虑其他单元的资讯--consider the neighbor
        
    - self attention--consider the whole sequence--out  
        <img src="../../_resources/dotproduct.png" alt="dotproduct.png" width="723" height="568" class="jop-noMdConv">
        
        > 全连接层专注处理某个位置的资讯，自注意力机制处理整个sequence的资讯，二者交替使用  
        > 如何考虑整个$a^1,a^2,a^3,a^4$ sequence: 相似性$\alpha$衡量,又叫 attention score  
        > 法一：Dot-product: $q=W^q\times a^1,k=W^k\times a^2 ,\alpha =q\cdot k$  
        > 法二： Additive: $q+k\rightarrow tanh\rightarrow W\rightarrow \alpha$
        
        - 逐元素$q^i=W^qa^i$,矩阵表示 $Q=W^qI$,$I$,input
            
            > 类似地 $K=W^kI$,$V=W^kI$  
            > $Q=[q^1,q^2,q^3,q^4]^T,K=[k^1,k^2,k^3,k^4]^T,V=[v1,v2,v3,v4]^T,I=[a^1,a^2,a^3,a^4]^T$
            
        - $A=QK^T$, $A'=softmax(A)$,$A'$, Attention Matrix
            > 注意力分数矩阵
            > $A=[\alpha_{1,:},\alpha_{2,:},\alpha_{3,:},\alpha_{4,:}]^T$
            
        - $O=A'V$ $O$,output
            > 归一化注意力分数矩阵
            > $O=[b^1,b^2,b^3,b^4]^T$
            
    - $Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt(d_k)})V$
        
        - 需要学习的参数：$W^q,W^k,W^v$
        - 这里V是$1\times 4$列向量，softmax得到的是$4\times 4$的 $\alpha'$ 矩阵
            - 此处softmax得到的矩阵，每一行为一个vector 和另外的vector的注意力得分
        - 也可写作：$Attention(Q,K,V)=V^Tsoftmax(\frac{KQ^T}{\sqrt(d_k)})$
            - 此处softmax得到的矩阵，每一列为一个vector 和另外的vector的注意力得分
    - Multi-head Self-attention (self-attention进阶版)
        
        - <img src="../../_resources/2head-attention.png" alt="2head-attention.png" width="492" height="524" class="jop-noMdConv">
        - 考虑到 Different types of relevance
        - 相较于自注意力机制，多头注意力机制head数目对应参数$W^q,W^k,W^v$的组数
            
            > 相当于多头注意力机制并行运行多个自注意力层并综合其结果  
            > 多头注意力机制通过不同组参数，捕捉输入序列在不同子空间中的信息  
            > 可以更好地处理复杂的语义关系，提高在NLP任务中的性能
            
    - Positional Encoding
        

	<img src="../../_resources/position_information.png" alt="position_information.png" width="240" height="185">


    - 考虑到位置的资讯
        
        > Each position has a unique positional vector $e^i$
        
    - 可以手工设计 hand-crafted
        
        > Compare position representation methods.有不同的方法，暂未有最有效  
        > 如Position embedding, RNN
        
- 应用
    
    - Self-attention for speech: 部分注意力机制 Truncated Self -attention
    - Self-attention for image: Self-Attention GAN; DEtection Transformer(DETR)
    - Self-attention for graph: 仅计算两连接的顶点间相关性矩阵
        - 注意：self-attention + graph = 一种 GNN （Graph Netural Network)
- Self-attention v.s. CNN
    
    - CNN: 在感受野作用的简化Self-attention，训练数据较少时表现相对更好
    - Self-attention: 感受野可学习，需要更多的训练数据
- Self-attention v.s. RNN
    
    - RNN,即使双向RNN可以考虑上下文，存在memory中难以解释,不能平行处理
    - Self-attention: 可平行处理，且上下文关联易解释，运算效率更高


* * *

## 李宏毅 transformer

- 编码器 Encoder
    
    - 捕捉输入序列的语义信息
    - 工作原理
    
    > 输入词汇通过嵌入层（Embedding Layer) 转换为固定向量表示  
    > 经多个自注意力层和前馈神经网络，捕捉词汇间依赖关系和语义信息
    
    - 缺陷：无法处理变长输入序列
    
    > 处理不同长度需要进行阶段或填充，可能引入误差
    

* * *

## pipelines

- NLP任务划分(transformer):
    - [ ] 文本分类：情感分析；句子对关系判断
    - [ ] 对文本中词语进行分类：词性标注（POS)，命名实体识别(NER)
    - [ ] 文本生成: 填充预设的模板(prompt)，预测文本中被遮掩掉(masker)
    - [ ] 从文本中抽取答案
    - [ ] 根据输入文本生成新的句子：文本翻译，自动摘要
- `pipeline()` transformer库基本函数
    - 封装了预训练模型前处理和后处理环节
        - [ ] `feature-extraction` 获得文本向量化表示
        - [ ] `fill_mask` 填充被遮盖的词、片段
        - [ ] `ner` 命名实体识别
        - [ ] `question-answering` 自动问答
        - [ ] `sentiment-analysis` 情感分析
        - [ ] `summarization` 自动摘要
        - [ ] `text-generation` 文本生成
        - [ ] `translation` 机器翻译
        - [ ] `zero-shot-classification` 零训练样本分类
- pipline运行机制
    - 遇到keras问题