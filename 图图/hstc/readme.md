# 深度学习对结构化产品定价的尝试
>王鹏
>>2018年7-8月

***                    
## 第一部分：初期尝试

### 雪球产品的结构图

![xueqiu](https://github.com/qiyexue777/noise/blob/master/图图/hstc/pic/xueqiu.png?raw=true)

### 信息提取
**根据上述产品信息，可从中提取出定价所需的可数字化的参数：**
- 产品合约期/duration：单位按月，整型；
- 敲出观察频率/knock_out_freq：单位按日，整型；
- 敲入观察频率/knock_in_freq：单位按日，整型；
- 敲出界限/knock_out_barrier：浮点型；
- 敲入界限/knock_in_barrier：浮点型；
- 敲出票息/knock_out_rate：浮点型；
- 红利票息/bonus：浮点型；
- 期望收益率/u：浮点型；
- 波动率/vol：浮点型；
- 标的物初期价格/spot：浮点型；

由以上产品参数，可得到代表一份产品的参数列表：

**key_list = [duration, knock_out_freq, knock_in_freq, knock_out_barrier, knock_in_barrier, knock_out_rate, bonus，u，vol, spot]**

注：
- 在 *初期尝试* 的过程中，上述的产品参数列表为定价部分的输入值，形式固定，参数相对位置不可更改；
- 如果需要产品参数中的某一个或某些以随机变量的形式生成，只需要将相应位置的参数设成 ***variant***， 后面的定价部分会自动处理；
- 当各个位置的变量被取成 ***variant*** 后，取值范围分别是：
  - duration: 12-24个月；
  - knock_out_freq: 1-252天，随机取整数；
  - knock_in_freq: 同上；
  - knock_out_barrier: 0.9-1.2，0，10；
  - knock_in_barrier: 0.7-0.9， 0，10；
  - knock_out_rate: 0-0.2;
  - bonus: 0-0.2;
  - u: 0-0.3;
  - vol: 0-0.3;
  - spot: 以100为基准，上下浮动5倍标准差；

### 雪球类产品定价模块构建
在确定上述输入变量形式后，我们可以构建以下的 ***SnowBall*** 类，其产品类参数初始化如下：
```
class SnowBall:

    duration = None
    knock_out_freq = None
    knock_in_freq = None
    knock_out_barrier = None
    knock_in_barrier = None
    knock_out_rate = None
    bonus = None
    spot = None
    std_spot = 100.
    argument = 100.
    u = None
    sigma = None
    curve_num = 50000
    r = 0.1
    bussiness_day = 252
    nature_day = 360
    bussiness_month = 21
    nature_month = 30
    month_num = 12
    stdi = np.float(np.sqrt(1./bussiness_day))
    aver = 0.
    tolerance = 0.001
```
神经网络类参数初始化如下：
```
session = None
   X_train_holder = None
   Y_train_holder = None
   Y_ = None
   cost = None
   optimizer = None
   w1 = None
   w2 = None
   b1 = None
   b2 = None
   ratio = 0.95

   learning_rate = 0.01
   input_num = 100
   input_dimension = 10
   output_dimension = 1
   hiden_layer_dimension = 12
   iteration = 3000
   accuracy = None
```
初始化函数：
```
def __init__(self, key_list):
    self.duration = key_list[0]
    self.knock_out_freq = key_list[1]
    self.knock_in_freq = key_list[2]
    self.knock_out_barrier = key_list[3]
    self.knock_in_barrier = key_list[4]
    self.knock_out_rate = key_list[5]
    self.bonus = key_list[6]
    self.u = key_list[7]
    self.sigma = key_list[8]
    self.spot = key_list[9]


    # build neural network
    self.create_network()

    # tensorflow
    self.session = tf.Session()
```
#### 蒙特卡洛部分
首先， 应根据产品合约参数生成标的物价格曲线，代码如下：
```
def underlying_curve(self, s0, t, curve_number, mu, vol):
    first = np.ones(curve_number)*s0
    length = np.int(t*self.bussiness_month)
    Wt = np.random.normal(self.aver, self.stdi, [curve_number, length-1])
    curve = s0*np.exp(vol*Wt+(mu-vol**2/2)/self.bussiness_day).cumprod(1)
    curve = np.c_[first, curve]
    return curve
```
在得到标的物价格曲线后，分别判断产品在续存期内是否有敲出和敲入事件发生：
```
def knock_out(self, upper_barrier, s0, frequency, underlying_curve, t):

    for i in range(np.int(self.bussiness_month * t / frequency)):

        index = int((i+1)*frequency)
        if underlying_curve[index-1]> s0*upper_barrier:
            return i+1
    return False

def knock_in(self, lower_barrier, s0, frequency, underlying_curve, t):

    for i in range(int(self.bussiness_month * t / frequency)):
        index = int((i+1)*frequency)
        if underlying_curve[index-1]< s0*lower_barrier:
            return True
    return False
```
随后，可以得到在现有的收益计算规则下，由一条标的物价格曲线得到的合约的价格：
```
def price_of_one_curve(self, one_product, one_curve):

    num = self.knock_out(one_product[3], self.std_spot, one_product[1], one_curve, one_product[0])
    if num:
        expire = num*self.nature_month/float(self.nature_day)
        discount_factor = np.exp(-self.r*expire)
        return self.argument * (1 + one_product[5] * expire)*discount_factor
    else:
        if self.knock_in(one_product[4], self.std_spot, one_product[2], one_curve, one_product[0]):
            receive = min(one_curve[-1] / self.std_spot, 1.)
            expire = one_product[0]/float(self.month_num)
            discount_factor = np.exp(-self.r*expire)
            return self.argument * receive * expire *discount_factor
        else:
            expire = one_product[0]/float(self.month_num)
            discount_factor = np.exp(-self.r*expire)
            return self.argument * (1 + one_product[6] * expire)*discount_factor
```
最后，通过多条蒙特卡洛模拟得到的价格曲线，去均值得到合约的期望价值：
```
def pricing(self, one_product):

    price = 0.
    curve = self.underlying_curve(one_product[-1], one_product[0], self.curve_num, one_product[7], one_product[8])
    for i in curve:

        price += self.price_of_one_curve(one_product, i)
    return price/self.curve_num
```    
#### 深度学习部分
在完成传统蒙特卡洛模拟定价部分之后，训练神经网络的记忆池就有了来源：基本思路是先生成多份合约参数不定的产品矩阵，矩阵的每一行代表一份不同的合约，然后利用上一部分的蒙特卡洛模拟来分别对产品矩阵的每一行进行定价，价格便代表label，是训练过程中要逼近的目标。换句话说，训练的目的并不是为了得到理论中合约的正确价格，而是逼近蒙特卡洛模拟得到的价格，这里默认蒙特卡洛模拟得到的价格是真实价值。（此处最好有插图来表示产品矩阵和label）

##### 记忆池生成
```
def book(self):

    '''
    产品续存期，取值范围12-24个月，随机取值取整数，可重复；
    :return:
    '''
    if self.duration is 'variant':
        c1 = np.random.random_integers(12,24,[self.input_num*1])
    else:
        c1 = np.ones((self.input_num,),dtype=np.int)*self.duration
    '''
    产品敲入和敲出观察频率，按天为单位，取值范围1-252天，随机取整数，可重复；
    '''
    if self.knock_out_freq is 'variant':
        c2 = np.random.random_integers(1,252,[self.input_num*1])
    else:
        c2 = np.ones((self.input_num,),dtype=np.int)*self.knock_out_freq

    if self.knock_in_freq is 'variant':
        c3 = np.random.random_integers(1,252,[self.input_num*1])
    else:
        c3 = np.ones((self.input_num,),dtype=np.int)*self.knock_in_freq

    '''
    两个barrier一上一下，upper barrier是0.9-1.2之间的float，随机取值；lower barrier是0.7-0.9之间的float，随机取值，并且上下界都
    可取到0和10，为了达到这种效果，利用了random函数的边缘特征，当随机变量取到区间左边界的时候，自动替换成0或10；
    '''
    if self.knock_out_barrier is 'variant':

        c4 = np.random.uniform(0.9, 1.2, [self.input_num*1])
        c4 = np.where(c4 == 0.9, np.random.choice([0., 10.]), c4)

    else:
        c4 = np.ones((self.input_num,))*self.knock_out_barrier

    if self.knock_in_barrier is 'variant':
        c5 = np.random.uniform(0.7, 0.9, [self.input_num*1])
        c5 = np.where(c5 == 0.7, np.random.choice([0., 10.]), c5)

    else:
        c5 = np.ones((self.input_num,))*self.knock_in_barrier

    '''
    敲出票息和红利都取到0-0.2；
    '''
    if self.knock_out_rate is 'variant':
        c6 = np.random.uniform(0,0.2,[self.input_num*1])
    else:
        c6 = np.ones((self.input_num,))*self.knock_out_rate

    if self.bonus is 'variant':
        c7 = np.random.uniform(0,0.2,[self.input_num*1])
    else:
        c7 = np.ones((self.input_num,))*self.bonus

    '''
    波动率和期望收益率都取到0-0.3；
    '''
    if self.u is 'variant':
        c8 = np.random.uniform(0., 0.3, [self.input_num*1])
    else:
        c8 = np.ones((self.input_num,))*self.u

    if self.sigma is 'variant':
        c9 = np.random.uniform(0., 0.3, [self.input_num*1])
    else:
        c9 = np.ones((self.input_num,))*self.sigma

    '''
    spot价格以100为基准，上下浮动5*std；

    '''
    if self.spot is 'variant':
        c10 = np.random.uniform(self.std_spot-5*self.stdi,self.std_spot+5*self.stdi,[self.input_num*1])
    else:
        c10 = np.ones((self.input_num,))*self.spot

    benchmark = np.c_[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10]
    return  benchmark
```
根据 ***信息提取*** 部分的讨论，book函数通过处理输入参数，生成一个benchmark矩阵：该矩阵的size为m x n，其中n的值为10，即产品合约中参数的总个数，m为合约的总份数，可由我们自行确定。

当然，只有产品的benchmark还不够，我们还需要每份合约对应的价格，这时需要label函数：
```
def label(self, bench):

    holder = []
    clock = 1

    time_total = 0.

    for i in bench:
        start = ti.time()

        holder.append(self.pricing(i))

        end = ti.time()
        consum = end - start
        time_total += consum

        if clock%100 is 0:
            print time_total
        clock += 1
    print 'average time consumption: '+str(time_total/self.input_num)
    return np.array(holder)
```
在调用label函数时，我们将book函数得到的benchmark矩阵传入其中，得到相应的label矩阵并返回。

到这一步，记忆池就已生成，也就是我们本次要利用的训练集和测试集。

#### 神经网初始化

利用create_network函数初始化placeholder，weight和bias并结合计算关系得到cost和optimizer。
```
def create_network(self):
    self.X_train_holder = tf.placeholder(shape=[int(self.input_num*self.ratio), self.input_dimension], dtype= tf.float32)
    self.Y_train_holder = tf.placeholder(shape=[int(self.input_num*self.ratio), self.output_dimension], dtype= tf.float32)

    self.w1 = tf.random_uniform(shape=[self.input_dimension, self.hiden_layer_dimension], minval=-0.05
    ,maxval=0.05, dtype=tf.float32, name='weight1')
    self.b1 = tf.Variable(tf.zeros([1, self.hiden_layer_dimension]), dtype= tf.float32)
    l1 = tf.nn.relu(tf.matmul(self.X_train_holder, self.w1) + self.b1)

    self.w2 = tf.random_uniform(shape=[self.hiden_layer_dimension, self.output_dimension], minval=-0.05
    ,maxval=0.05, dtype=tf.float32, name='weight2')
    self.b2 = tf.Variable(tf.zeros([1, self.output_dimension])+ 0.1, dtype= tf.float32)
    self.Y_ = tf.nn.relu(tf.matmul(l1, self.w2) + self.b2)

    self.cost = tf.reduce_mean(tf.square(self.Y_train_holder - self.Y_)) / 2.
    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
```
开始训练进程：
```
@property
    def train_test(self):

        benchmarks = self.book()

        labels = self.label(benchmarks)

        train_set = benchmarks[0:int(self.input_num*self.ratio)]
        test_set = benchmarks[int(self.input_num*self.ratio):self.input_num]

        y_train = labels[0:int(self.input_num*self.ratio)].reshape([int(self.input_num*self.ratio),1])
        y_test = labels[int(self.input_num*self.ratio):self.input_num].reshape(int(self.input_num-int(self.input_num*self.ratio)),1)

        self.session.run(tf.global_variables_initializer())


        for rand in range(self.iteration):

            self.session.run(self.optimizer,feed_dict={self.X_train_holder: train_set, self.Y_train_holder: y_train})

        weight1, weight2, bias1, bias2 = self.session.run([self.w1, self.w2, self.b1, self.b2])

        hiden_layer = tf.nn.relu(tf.matmul(tf.cast(test_set, tf.float32), self.w1)+ self.b1)

        y_predict = tf.nn.relu(tf.matmul(hiden_layer, self.w2)+ self.b2)

        count = tf.divide(np.abs(y_predict-y_test),y_test)

        mat = count - self.tolerance
        result = self.session.run(mat)

        return weight1, weight2, bias1, bias2, result
```
利用梯度下降算法最小化cost，将得到的参数矩阵返回。

需要留意的是，该train_test函数内部已经包含了测试模块儿，即从生成的记忆池中整体取出**百分之五**的合约，来验证预测价格和真实label之间的差距，返回到result矩阵，该矩阵代表误差率，其中每个元素的定义为：

$item := \frac{\vert price_p - price \vert}{price} - tolerance$

可见， item的值越小越好，下图为部分测试结果：
