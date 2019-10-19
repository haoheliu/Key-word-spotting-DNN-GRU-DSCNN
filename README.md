
I utilized DNN(3*128 and 6*512),GRU,DSCNN to do key word spotting task


# 1. Tasks - DO KEY WORD SPOTTING

Feature representation &rarr; Model &rarr; Posterior handling &rarr; Evaluation on test set 

- Key word: *HELLOW XIAOGUA*
- Training set: 1642 positive example, 9383 positive example
- Test set: 559 positive example, 3453 negative example
- Feature extraction rate: computed every 10ms over a window of 25ms
- Feature shape: (frameNumbers,40)
- Performance metrics: False-reject and False positive rate

# 2. Feature representation

In order to mimic the mechanism of human ear, we stack the nearby frames on labeling a specify frames, as the below picture show:

![Feature stack](https://ranchofromxgd.github.io/assets/2019-10-19-18-18-34.png)

There are two types of feature stacking, one is stack vertically(Type2) and another is stack horizontally(Type1).Type1 can be consider as an image so is easy to perform conv-op and used as input of DSCNN. Type2 is a single vector, so can be used as the input of GRU or DNN.

As for the value of m and n

| model      | n    | m    |
| ---------- | ---- | ---- |
| DNN        | 30   | 10   |
| GRU, DSCNN | 15   | 5    |

Then lets take a look at the models I used.

# 3. Models

## 3.1. DNN

DNN with 3(or 6) layers and each layer have 128(or 512) units

<figure class="half">
    <img src="https://ranchofromxgd.github.io/assets/2019-10-19-17-44-20.png" width="400" height="530" >
    <img src="https://ranchofromxgd.github.io/assets/2019-10-19-17-46-35.png" width="300" height="530" >
</figure>

### 3.1.1. Loss function

- Step 1: Softmax

$$
softmax(x)_i = \frac{exp(x_i)}{\sum_jexp(x_j)}
$$

- Step 2: Cross entropy

$$
Output_{y_0}(y) = -\sum_iy_o^{<i>}log(y^{<i>}) 
$$

$$
y_0 :true label
$$

$$
y: output logits
$$

### 3.1.2. Hyper parameter

```python
    trainBatchSize = 100
    testBatchSize = 100
    leftFrames = 30
    rightFrames = 10
    learningRate = 0.00001
    decay_rate = 0.8
    numEpochs = 5
    w_smooth = 3
    w_max = 30
```

- Shuffle train and test data every epoch
- Learning rate exponential decay

## 3.2. GRU


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

<figure class="half">
    <img src = "https://ranchofromxgd.github.io/assets/2019-10-19-18-47-14.png" width="380" height="530">
    <img src = "https://ranchofromxgd.github.io/assets/2019-10-19-18-48-33.png" width="380" height="530" >
</figure>

### 3.2.1. Loss function

- tensorflow.contrib.seq2seq.sequence_loss

Apply cross-entropy loss between each element and its label in a sequence. The sequence length is not fixed, so we need to input a mask as a filter

### 3.2.2. Hyper parameter

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

## 3.3. DSCNN

| Layer                       | Parameter                                                |
| --------------------------- | -------------------------------------------------------- |
| 1:Conv+batchNorm            | kernel-(10,4), y_stride-2, x_stride-1, outputfeature-172 |
| 2:DS-Conv                   | kernel-(3,3), y_stride-2, x_stride-2, outputfeature-172  |
| 3:DS-Conv                   | kernel-(3,3), y_stride-1, x_stride-1, outputfeature-172  |
| 4:DS-Conv                   | kernel-(3,3), y_stride-1, x_stride-1, outputfeature-172  |
| 5:DS-Conv                   | kernel-(3,3), y_stride-1, x_stride-1, outputfeature-172  |
| 6:AvgPooling+FullyConnected | (None,3)                                                 |

<figure class="half">
    <img src = "https://ranchofromxgd.github.io/assets/2019-10-19-18-44-12.png" width="400" height="560">
    <img src = "https://ranchofromxgd.github.io/assets/2019-10-19-18-44-45.png" width="380" height="560">
</figure>


Note: DS-Conv stands for [Depthwise Separatable Convolution](https://blog.csdn.net/tintinetmilou/article/details/81607721), which include a depthwise convolution, batchnorm, a pointwise convolution and batchnorm.

### 3.3.1. Loss function

The same as DNN

### 3.3.2. Hyper parameter

```python
    trainBatchSize = 10
    testBatchSize = 10
    leftFrames = 15
    rightFrames = 5
    shuffle = True
    learningRate = 0.000002
    decay_rate = 0.895
    numEpochs = 60
    w_smooth = 5
    w_max = 70
```

# 4. Posterior handlng

For the output of the model, take following two steps:

- Smoothing  

  $$
  p^{'}_{ij}=\frac{1}{j-h_{smooth}+1}\sum_{k=h_{smooth}}^{j}p_{ik}
  $$

  $$
  h_{smooth}=max\{1,j-w_{smooth}+1\}
  $$

  Here i's value can be {0,1,2}, which stand for {filler,keyword1,keyword2}

- Calculate Confidence

  $$
  confidence = \sum^{n-1}_{i=1}max_{h_{max<=k<=j}}p_{ik}^{'}
  $$

Then I select the max value in confidence as the score of a specific data

# 5. Result

First let's do a brief compare to these four model

### 5.1. Parameter number

| MODEL     | PARAMETERS                                               |
| --------- | -------------------------------------------------------- |
| 3-128 DNN | (1640,128)+2*(128,128)+2*(128,)+(128,3)+(3,) = 243,459   |
| 6-512 DNN | (1640,512)+5*(512,512)+6*(512,)+(512,3)+(3,) = 2,155,011 |
| GRU_128   | (968,256)+(256,)+(968,128)+(128,)+(128,3)+(3,) = 372,483 |
| DSCNN     | 135,023                                                  |


Note: leftFrames and rightFrames refer to the frames used during frame stacking

### 5.2. Performance

- DSCNN > GRU > DNN_512_6 > DNN_128_3

![ROC](https://ranchofromxgd.github.io/assets/2019-10-18-17-04-21.png)

### 5.3. Visualization

- In order to examine the performance of model and make the debugging easier, I made the following visualization:
  ![compare](https://ranchofromxgd.github.io/assets/2019-10-19-19-29-33.png)

- For top to bottom, I visualized:   

1. **Wave form:** Plot the raw .wav file
2. **Desired label:** Where 0 stand for 'filler',1 stand for 'hello',2 stand for 'xiaogua'
3. **Modeloutput_label_0**: The probability of label0 (filler) in model output
4. **Modeloutput_label_1_2**: The probability of label1(hello) and label2(xiaogua) in model output
5. **Confidence**: The value we get after posterior handling of model output

# 6. Problems

- I recorded these problems in my blog

1. [Loss value soar abnormally after one epoch](http://haoheliu.com/2019/10/10/Use-GRU-for-Key-word-spotting/#51-%E5%88%B0%E7%AC%AC%E4%BA%8C%E4%B8%AAepoch%E4%BB%A5%E5%90%8Eloss%E6%BF%80%E5%A2%9E)
2. [Smoothing and the calculation of confidence described in paper doesn't make sense in this project, so I improved it](http://haoheliu.com/2019/09/29/Procedure-and-Problems-I-have-during-the-build-of-DNN-key-word-spotting-system/#44-%E8%AE%BA%E6%96%87%E4%B8%AD%E6%9C%89%E5%93%AA%E4%BA%9B%E9%9C%80%E8%A6%81%E6%94%B9%E8%BF%9B%E7%9A%84%E5%9C%B0%E6%96%B9)
3. [Data transforming process is too slow and GPU-Util is low](http://haoheliu.com/2019/09/29/Procedure-and-Problems-I-have-during-the-build-of-DNN-key-word-spotting-system/#41-%E5%A6%82%E4%BD%95%E9%81%BF%E5%85%8Dio%E5%92%8C%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F%E8%BD%AC%E6%8D%A2%E6%88%90%E4%B8%BA%E8%AE%AD%E7%BB%83%E9%80%9F%E5%BA%A6%E7%9A%84bottle-neck-%E7%94%A8cpu%E5%92%8Cgpu-%E6%95%88%E7%8E%87%E7%9B%B8%E8%BF%91)
4. [The usage of tensorflow](http://haoheliu.com/2019/10/12/Tensorflow-Learning/)

# 7. Conclusions

- DSCNN:   
  It's the best one among these model, due to its light weight(less parameters),high concurrency and best performance. Firstly, DSCNN utilize the strength of CNN. It makes the kernel's weight params shared across different region, making model both robust and easy to train. Secondly, it lower the weight number even more comparing with CNN. By the application of depth-wise conv and point-wise conv, DSCNN is able to expand features efficiently and reduce params significantly.
- GRU:  
  It has moderate performace in this project. It's better than DNN from two aspect: weight sharing, hidden state. First, Weight sharing mechanism both shrink the size of model and enable the efficient use of params. Second, hidden state, which can be seen as embedding of prior frames, is also useful during the classification of current frame.
- DNN:  
  It is both cumbersome and incompetent. The advantage of this model, if it have one, may be its simplicity of realization.

# 8. Summary

- It takes me about seventeen days to accomplish this project, which is the first project in ASLP. I think I have learnt a lot during this period of time. First of all, I have know about the common procedure realizing of a speech recognition system. Secondly I dive more deeper in DL by carefully examination, debugging and realization of GRU,DNN and DSCNN. Third, I become more proficient in using python and tensorflow. Besides, I found this process really interesting and offer me with a sense of accomplishment. Now I'm ready for future's more challenging task!

# 9. Links

- [Github: Key-word-spotting-DNN-GRU-DSCNN](https://github.com/Ranchofromxgd/Key-word-spotting-DNN-GRU-DSCNN)
- [BLOG: Procedure and Problems I have during the build of DNN key word spotting system](http://haoheliu.com/2019/09/29/Procedure-and-Problems-I-have-during-the-build-of-DNN-key-word-spotting-system/#41-%E5%A6%82%E4%BD%95%E9%81%BF%E5%85%8Dio%E5%92%8C%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F%E8%BD%AC%E6%8D%A2%E6%88%90%E4%B8%BA%E8%AE%AD%E7%BB%83%E9%80%9F%E5%BA%A6%E7%9A%84bottle-neck-%E7%94%A8cpu%E5%92%8Cgpu-%E6%95%88%E7%8E%87%E7%9B%B8%E8%BF%91)
- [BLOG: Use GRU for Key word spotting](http://haoheliu.com/2019/10/10/Use-GRU-for-Key-word-spotting/)
