# Key-word-spotting-DNN
Tensorflow实现的DNN-KWS以及GRU-KWS系统
- ./models存储了训练好的效果最好的模型
- ./pickles储存了绘制ROC曲线的原始数据
- config.py 用来配置模型以及调试参数（全局）
- dataLoader.py 用来加载test和trian数据
- util.py 包含各种工具函数，包括特征格式转换，绘制曲线，可视化，数据统计等
- train.py 包含建图和训练的过程
- model.py 构建了DNN,GRU模型
- PosteriorHandling.py 实现了模型输出的后处理
- fbankreader3.py 用来读取fbank文件
- makeGif.py 用于将一个目录下的所有png图片合成为gif
- countParamNumber.py 用于数目前的模型有多少参数
 
# 一. 链接
 - [Github - Ranchofromxgd/Key-word-spotting-DNN-GRU](https://github.com/Ranchofromxgd/Key-word-spotting-DNN)
 - [GRU语音识别实现过程](http://haoheliu.com/2019/10/10/Use-GRU-for-Key-word-spotting/)
 - [DNN语音识别实现过程](http://haoheliu.com/2019/09/29/Procedure-and-Problems-I-have-during-the-build-of-DNN-key-word-spotting-system/)
 - [Paper - SMALL-FOOTPRINT KEYWORD SPOTTING USING DEEP NEURAL NETWORK](https://research.google.com/pubs/archive/42537.pdf)
 - [友情链接 - 好河流](haoheliu.com)

# 二.GRU语音识别实现过程
# 1. 概述
- 这个实验是基于上一个[DNN-KWS](http://haoheliu.com/2019/09/29/Procedure-and-Problems-I-have-during-the-build-of-DNN-key-word-spotting-system/)来做的，只不过换用了GRU模型，需要改变数据输入格式和损失函数等

- 实现使用的是tensorflow，在过程中遇到的问题记录在了这篇blog中：[Tensorflow 战术总结1](http://haoheliu.com/2019/10/12/Tensorflow-Learning/)

## 1.1. 链接
- [Github:  Key-word-spotting-DNN-GRU](https://github.com/Ranchofromxgd/Key-word-spotting-DNN-GRU)


# 2. 运行结果
## 2.1. ROC曲线
GRU和之前用DNN两个模型的对比

![Compare](https://ranchofromxgd.github.io/assets/2019-10-15-14-13-14.png)

## 2.2. 模型输出可视化
一个Gif图（需要等着加载一会）
- 从上到下依次是声音波形、期望的输出标签、模型输出的filler的置信度、模型输出的两个唤醒词的置信度，整体Confidence
- 可以看到在positive样例（标题中显示是P还是N）期望输出标签的位置，模型对两个唤醒词的置信度也相继会上升，对应的confidence也比较高。Negative样例则输出值都比较低。

![Compare_gif](https://ranchofromxgd.github.io/assets/compare.gif)

## 2.3. 最优参数
最终使用的最优参数为：
```python
    modelName = "GRU" # "GRU" "DNN_6_512" "DNN_3_128"
    lossFunc = "seqLoss" # "Paper" "crossEntropy"
    trainBatchSize = 16
    testBatchSize = 16
    leftFrames = 15
    shuffle = True
    rightFrames = 5
    learningRate = 0.001
    decay_rate = 0.895
    numEpochs = 60
    w_smooth = 5
    w_max = 70
```
- Loss & learningRate 曲线

![loss_lr](https://ranchofromxgd.github.io/assets/2019-10-15-14-19-58.png)


# 3. 模型结构
- 模型的整体结构如下  

![Overall](https://ranchofromxgd.github.io/assets/2019-10-15-14-12-24.png)

- 其中GRU模型如下：

$$
c^{<t>}_0 = tanh(W_c[c^{<t-1>},x^{t}]+b_c)
$$

$$
c^{<t>} = \Gamma_u*c^{<t>}_0 + (1-\Gamma_u)*c^{<t-1>}
$$

Update gate:

$$
\Gamma_u = \sigma(W_u[c^{<t-1>},x^{t}]+b_u)
$$

## 3.1. 参数个数
- 根据上述公式，GRU的trainable parameter有$W_c,W_u,b_c,b_u$, 形状分别为(968, 256)，256，(968, 128)，128，再加上一层DNN(128,3)，最终参数个数为372483
 
- 下边对主要的三个板块进行展开：
## 3.2. Placeholders

![Placeholder](https://ranchofromxgd.github.io/assets/2019-10-13-13-55-40.png)

## 3.3. GRU-DenseLayer

![model](https://ranchofromxgd.github.io/assets/2019-10-13-13-58-46.png)

## 3.4. Loss function and optimizer

![loss_function_and_optimizer](https://ranchofromxgd.github.io/assets/2019-10-13-14-05-03.png)


# 4. 实现细节
## 4.1. GRU模型定义
```python
def GRU(self,batch,length):
    with tf.variable_scope("GRU"):
        # 先定义一个cell，输出为128维
        cell = tf.contrib.rnn.GRUCell(num_units=128,name = "gru_cell")
        outputs, _ = tf.nn.dynamic_rnn(
            cell=cell, 
            dtype=tf.float32,
            sequence_length=length,
            inputs=batch, 
            time_major= False) # 如果是True的话输入的第一维是timestep，否则第一维是batchsize
        outputs = tf.layers.dense(outputs, 3,name="dense_output")
    return outputs
```
注意这里对输入数据的维度要求
- **time_major:**
如果time_major=True的话，输入的input形状应该是[Time_Step,batch_size,feature_size]  
如果time_major=False的话，输入的input形状应该是[batch_size,Time_Step,feature_size]
- **sequence_length:**  
对应的是batch中每一个数据的有效长度

## 4.2. GRU输入数据
经过统计，输入数据中最多为1249帧，在构造数据的时候就按照1300帧来对所有帧进行padding，在计算loss的时候，根据各个数据的实际帧长度生成一个mask，与loss点积后获得实际的的loss
<hr>
**10.14 UPDATE** 

发现全局padding效率比较低，后边就按照每个batch最长的序列来padding

## 4.3. 损失函数Sequence loss
- 输入一个tensor的列表，对每条数据的每一个时间片都计算一个交叉熵损失，一条数据中各个时间片的损失加起来取平均，各个数据之间也取平均  
- 最初使用sequence_loss的时候直接调的包，结果一直提示tensor不能迭代，查看源码以后才发现输入竟然要求是列表，因为用到了zip操作，我也只得将tensor进行分片然后放入list
```python
    def sequence_loss(self,
                      logits,
                      targets,
                      weights,
                      average_across_timesteps=True,
                      average_across_batch=True,
                      softmax_loss_function=None,
                      name=None):
        with tf.name_scope(name, "sequence_loss", logits + targets + weights):
            cost = tf.reduce_sum(
                self.sequence_loss_by_example(
                    logits,
                    targets,
                    weights,
                    average_across_timesteps=average_across_timesteps,
                    softmax_loss_function=softmax_loss_function))
            if average_across_batch:
                batch_size = tf.shape(targets[0])[0]
                return cost / tf.cast(batch_size, cost.dtype)
            else:
                return cost

    def sequence_loss_by_example(self,
                                 logits,
                                 targets,
                                 weights,
                                 average_across_timesteps=True,
                                 softmax_loss_function=None,
                                 name=None):
        # 此三者都是列表，长度都应该相同
        if len(targets) != len(logits) or len(weights) != len(logits):
            raise ValueError("Lengths of logits, weights, and targets must be the same "
                             "%d, %d, %d." % (len(logits), len(weights), len(targets)))
        with tf.name_scope(name, "sequence_loss_by_example",
                           logits + targets + weights):
            log_perp_list = []
            # 计算每个时间片的损失
            for logit, target, weight in zip(logits, targets, weights):
                if softmax_loss_function is None:
                    # 默认使用sparse
                    target = tf.reshape(target, [-1])
                    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=target, logits=logit)
                else:
                    crossent = softmax_loss_function(labels=target, logits=logit)
                log_perp_list.append(crossent * weight)
            # 把各个时间片的损失加起来
            log_perps = tf.add_n(log_perp_list)
            # 对各个时间片的损失求平均数
            if average_across_timesteps:
                total_size = tf.add_n(weights)
                total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
                log_perps /= total_size
        return log_perps
```
<hr>
**10.15 UPDATE**  

上边的源码是老旧的，可以直接使用tensorflow.contrib.seq2seq.sequence_loss来计算，不需要转化为list

# 5. 遇到的部分问题记录
## 5.1. 到第二个epoch以后loss激增
如图：

![loss_problem](https://ranchofromxgd.github.io/assets/2019-10-14-18-00-13.png)

是因为训练数据在第一个epoch忘了shuffle了，导致刚开始用来训练的全是正样本，完了才是负样本，到第二个epoch shuffle之后训练集的分布就s会就会变动比较大，影响准确度

<hr>
后记：这次实现的比较慢，前后花了快四天的时间，其中两天在调模型。经过这次调试，我懂得了shuffle以及各种trick的重要性。以后记住了在实现模型的时候能调API的就尽量不要自己尝试实现，总不如API来得效果好。今后调试也要更多的使用tensorboard，真是调试利器。PDB也要开始学着使用了。


# 三. DNN实现过程
## 1. Overall framework
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019092921052557.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
## 2. Time line
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929212513365.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
## 3. Experiment result

 Model capacity: 
 1. 3*128 DNN with Relu activation
 2. 6*512 DNN with Relu activation

### 3.1 ROC curve

Evaluated on 559 positive test data and 3453 negative test data:

![Roc](http://haoheliu.com/assets/2019-10-07-00-28-30.png)

### 3.2 Model output v.s Raw *.WAV* data (List only a few)
#### 3.2.1 Positive test data
##### (1) positive_00001
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929213338881.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
##### (2) positive_00002
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929213518770.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
##### (3) positive_00003
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929213535142.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
#### 3.2.2 Negative test data
##### (1) negative_00001
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929213723705.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
#####  (2) negative_00002
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929213758253.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
#####  (3) negative_00003
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929213823355.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
## 4. Problem encountered
### 4.1 如何避免IO和数据格式转换成为训练速度的bottle neck （用CPU和GPU 效率相近）
初期训练时用CPU和GPU速度相近，故而怀疑时其他地方消耗了时间。在初期测试代码性能时得到如下结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190926170320974.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
可以发现其中fbankTransform只调用了72次，但是在程序总运行时间中占到了80%以上
这个函数主要进行拼帧操作，其中主要用到 concatenate/ append/ vstack ,但是这些操作都是比较慢的，由于每次扩展numpy 数组都要新开辟一块内存，类似于Java里边的Strring，详见下边这篇博客：
[Python numpy数组扩展效率问题 ](https://blog.csdn.net/jmy5945hh/article/details/38091485)
为了解决这个问题，我考虑了如下两个方案：
- 将41帧按顺序append这个操作改为将41帧的矩阵直接reshape，一步搞定
- 将转换后的数据存入磁盘，在下次使用时无需转换
在尝试了第一种方法之后，发现效率提高了近一倍，然而还是非常慢，故而直接使用第二个方案：
将所有test和train data处理后存为数组到磁盘 "./OfflineData"：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929220049599.png)

转换后的数据总共有51G，但是这一级Cache使得训练过程加速了四倍！

### 4.2 如何解决某一类概率输出dominant的问题
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929220700211.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
最初模型对每一类预测的概率输出如上图（红线时filler，蓝线为hello，绿线为xiao gua）
刚开始感觉非常费解，为什么随着训练的进行，模型倾向于把所有的frames都认为是filler？
- 最初认为是数据正负样本不均导致的，在减少了负样本后没有缓解，故排除
- 在百思不得其解后翻阅了tensorflow的document，发现新大陆！从document的叙述中可以看出，tensoflow特别强调了logits需要unscaled logits，不需要提前做softmax。而我的模型在最后一步使用了softmax，故而导致结果错误。但是原理是什么呢？
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929221002855.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
首先想一下Softmax的原理：
假设有一个数组$V$,$V_i$表示其中第$i$个元素，那么这个元素的softmax值为

$$S_i = \frac{e^i}{\sum_je^j}$$

可以看出softmax是对$[-inf,+inf]$空间的一个非线性映射，而且对小于0的数据空间进行了压缩
而从3.2.1节图片可以看出，模型对hello和xiaogua的输出值大部分都小于零，所以经过指数运算后输出值就被压缩，而且还被softmax了两次，始得模型的输出准确率大大下降

### 4.3 如何高效的实现论文中提到的loss function
论文中公式将每一帧正确label的概率输出提取了出来

然而在计算图中tensor是不允许像数组一样迭代和提取元素的
- 解决方案：
根据label构建一个one-hot矩阵，之后依次使用向量点积将元素取出：
```python
	index = tf.argmax(targets, axis=1,name="argmax_Convert3_1tolabel")
	oneHot = tf.cast(tf.one_hot(index, 3, 1, 0), dtype=tf.float32)
	output = tf.log(tf.reduce_sum(modelOutput * oneHot, 1,name="Add_all"))
	output = -tf.reduce_sum(output)
```
### 4.4 论文中有哪些需要改进的地方
- Posterior handling 使用的公式不太符合一般情况
最初我按照论文中提供的$Confidence$计算公式：

  $$confidence = \sqrt[n-1]{\prod^{n-1}_{i=1}max_{h_{max<=k<=j}}p_{ik}^{'}}$$

  由3.2.1节可知，模型输出的probability是在$[-inf,+inf]$这个范围内的，有正有负，如果两个都是负的话，相乘之后符号就被抵消掉了，反而可能比两个都是正的数乘起来要大，这显然不合理。如下图所示：
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929224639763.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
$Confidence$理应是呈上升趋势，而这里却变为下降，由于后处理设计的漏洞

    为了克服符号造成的影响，在这里 $Confidence$ 的计算按下边的公式来:

  $$confidence = \sum^{n-1}_{i=1}max_{h_{max<=k<=j}}p_{ik}^{'}$$

- 平滑范围不太合适
1. 最初平滑值$w_{smooth}$和$w_{max}$我分别取的是30和100(论文里边的取值)，得到类似于下方的结果：
*正样本：*
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929225323654.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
*负样本*：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929225456112.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
由于平滑的范围过大，导致结果过于平滑，和负样本没有很大的区分度
2. 使用$w_{smooth}$和$w_{max}$分别取3和10：
*正样本：*
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190929225817650.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
*负样本：*
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019092922585628.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzIzNTU5,size_16,color_FFFFFF,t_70)
可以发现对confidence的衡量变得精准的多
## 5.实现细节
- 获取训练数据预分配空间，以避免大量append占用内存且效率低
```python
class dataLoader:
    def __init__(self):
    
        ...
        
        # 设置预分配空间大小
        self.maxTestCacheSize = Config.testBatchSize * Config.maximumFrameNumbers
        self.maxTrainCacheSize = Config.trainBatchSize * Config.maximumFrameNumbers
        # 预分配空间
        self.trainData = {
            "data": np.empty(shape=[self.maxTrainCacheSize,(Config.leftFrames+Config.rightFrames+1)*40]),
            "label": np.empty(shape=[self.maxTrainCacheSize,3])
        }
        self.testData = {
            "data": [],
            "label": []
        }
			
		...

    # Get a batch of positive training example
    def getTrainNextBatch(self):
        # Reset
        self.trainData = {
            "data": np.empty(shape=[self.maxTrainCacheSize,(Config.leftFrames+Config.rightFrames+1)*40]),
            "label": np.empty(shape=[self.maxTrainCacheSize,3])
        }
        counter,currentRow = 0,0
        # Report
        if(self.currentTrainDataFile % 1000 == 0):
            print(str(self.currentTrainDataFile)+" training files finished!")
        for i in range(Config.trainBatchSize):
            if(self.currentTrainDataFile >= len(self.trainDataFiles)):
                self.currentTrainDataFile = 0 # repeat the hole dataset again
                Config.numEpochs -= 1
                if(Config.shuffle == True):
                    print("Shuffle training data ...")
                    random.shuffle(self.trainDataFiles)
                return np.empty(shape=[0]),np.empty(shape=[0])
            fname = self.util.splitFileName(self.trainDataFiles[self.currentTrainDataFile])
            try:
                result = np.load(Config.offlineDataPath+fname+"_data.npy")
                label = np.load(Config.offlineDataPath+fname+"_label.npy")
            except:
                print("Error while reading file: "+fname)
                self.currentTrainDataFile += 1
                continue
            self.currentTrainDataFile += 1
           	# 每次对一行进行一次复制操作
            for data,label in zip(result,label):
                self.trainData['data'][currentRow] = data
                self.trainData['label'][currentRow] = label
                currentRow += 1
        # 去除预分配空间中不需要的部分
        self.trainData['data'] = self.trainData['data'][:currentRow]
        self.trainData['label'] = self.trainData['label'][:currentRow]
        return self.trainData['data'],self.trainData['label']

    def getSingleTestData(self,fPath = None):
        if(self.currentTestDataFile >= len(self.testDataFiles)-0):
            self.currentTestDataFile = 0
            random.shuffle(self.testDataFiles)
            return [],[]
        if(not fPath == None):
            fname = self.util.splitFileName(fPath)
        else:
            fname = self.util.splitFileName(self.testDataFiles[self.currentTestDataFile])
        try:
            result = np.load(Config.offlineDataPath + fname + "_data.npy")
            label = np.load(Config.offlineDataPath + fname + "_label.npy")
        except:
            print("Error while reading file: " + fname)
            return [],[]
        testData = {
            "data": np.zeros(shape=[result.shape[0], (Config.leftFrames + Config.rightFrames + 1) * 40]),
            "label": None
        }
        currentRow = 0
        self.currentTestDataFile += 1
        # 每次往缓冲区填充一行的数据
        for data, label in zip(result, label):
            testData['data'][currentRow] = data
            currentRow += 1
        type = fname.strip().split('_')[0]
        if(type == 'positive'):
            testData['label'] = 1
        elif(type == 'negative'):
            testData['label'] = 0
        else:
            raise ValueError("File should either be positive or negative")
        return testData['data'],testData['label']
```

- 特征拼接的实现：
```python
    # Combine each frame's feature with left and right frames'
    def fbankTransform(self,fPath = "positive_00011.fbank",save = True,test = True):
        raw = fbankreader3.HTKFeat_read(fPath)
        raw = raw.getall().tolist()
        frameLength = len(raw)
        result = np.empty(shape=[0,(Config.leftFrames+Config.rightFrames+1)*40])
        fname = self.splitFileName(fPath)  # e.g. positive_00011
        label = np.empty(shape=[0,3])
        raw = np.array([raw[0]]*Config.leftFrames+raw+[raw[-1]]*Config.rightFrames) # This trick can make algorithm more efficient
        for i in range(0,frameLength):
            base = i + Config.leftFrames
            temp = raw[i:base + Config.rightFrames + 1].reshape((1,(Config.leftFrames+Config.rightFrames+1)*40)) # This can outperform np.append
            result = np.concatenate((result, temp),axis=0)
        for i in range(0, len(result)):
            if(self.isFirstKeyWord(fname,i)):
                label = np.append(label,np.array([[0,1,0]]),axis=0)
            elif(self.isSecondKeyWord(fname,i)):
                label = np.append(label,np.array([[0,0,1]]),axis=0)
            else:
                label = np.append(label, np.array([[1, 0, 0]]),axis=0)
        if(save == True):
                np.save(Config.offlineDataPath + fname+"_data.npy",result)
                np.save(Config.offlineDataPath + fname + "_label.npy", label)
        return result,label
```

- 训练过程: 

```python
print("Construct model...")
dnnModel = Model()
print("Construct dataLoader...")
dataloader = dataLoader()
batch = tf.placeholder(tf.float32,shape = (None,40*(Config.leftFrames+Config.rightFrames+1)),name = 'batch_input')
label = tf.placeholder(tf.float32,shape = (None,3),name="label_input")
Loss,_ = dnnModel.lossFunc_CrossEntropy(batch, label)
saver = tf.train.Saver()
print("Construct optimizer...")

with tf.name_scope("modelOptimizer"):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(Config.learningRate,
                                               global_step=global_step,
                                               decay_steps=int(10000/Config.trainBatchSize),
                                               decay_rate=Config.decay_rate)
    trainStep = tf.train.GradientDescentOptimizer(learning_rate=Config.learningRate,name="gradient_optimizer").minimize(Loss,global_step=global_step)

print("Start Training Session...")
with tf.Session(config=config) as sess:
    print("Initialize variables...")
    sess.run(tf.global_variables_initializer())
    # writer = tf.summary.FileWriter("./log", tf.get_default_graph())
    # writer.close()
    while(not Config.numEpochs == 0):
        currentEpoch = Config.numEpochs
        if(Config.numEpochs % 1 == 0):
            print("Start testing... ", end="")
            dataloader.visualizaPositiveDataFiles(dataloader.testDataFiles,sess,dnnModel.model)

        saver.save(sess,"./model/model.ckpt")
        while(1):
            batchTrain,labelTrain = dataloader.getTrainNextBatch() # Get a batch of data
            batchTrain, labelTrain = shuffle(batchTrain,labelTrain)
            if(not currentEpoch == Config.numEpochs):
                break
            sess.run(trainStep,feed_dict={batch:batchTrain,label:labelTrain})
        print("[EPOCH " + str(totalEpoches - Config.numEpochs+1), "]", "lr: ", sess.run(learning_rate))
```
