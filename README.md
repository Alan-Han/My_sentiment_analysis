# My_sentiment_analysis
本项目数据集内容如下：
reviews.txt - 含有 25000 条IMDB影评  
labels.txt - 针对 reviews.txt 中的影评的 positive/negative 情感标签
首先对数据进行预处理、编码，然后通过embed得到词向量，建立LSTM网络进行训练，MSE作为loss函数，最终测试集上正确率达到82%
