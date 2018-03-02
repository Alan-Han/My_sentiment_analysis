# My_sentiment_analysis
本项目对IMDB影评进行情感分析，输出positive/negative
inputs文件夹中存放有25000条IMDB影评reviews及对应情感标签labels的txt格式文件

## 1.	运行环境 

ios 10.13  python 3.5.4  tensorflow 1.2.0


## 2. 数据集预处理：

运行主程序main.py后先对数据进行预处理：
(1)将reviews去除标点符号，并移除长度为0的异常review数据

(2)建立word的字典并对words及labels进行编码

(3)设置长度为200的feature向量作为训练输入数据

(4)将训练数据的10%作为验证集、10%作为测试集，以.p文件格式存放在preprocess_data文件夹中


## 3. 训练模型

(1)运行train()后开始训练模型，模型为1层含有256个LSTM的RNN网络，设置optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)，训练10次

(2)词向量长度embed_size = 300，学习率为0.001


## 4. 测试模型

主程序运行test_model()开始测试，10个epochs后最终准确率达到82%
