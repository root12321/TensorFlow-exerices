import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
def compute_accuracy(v_xs,v_ys):
    global prediction#定义全局变量
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})#
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))#将实际值和预测值进行比较，返回Bool数据类型
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#将上面的bool类型转为float，求得矩阵中所有元素的平均值
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})#运行得到上面的平均值，这个值越大说明预测的越准确，因为都是0-1类型，所以平均值不超过1
    return result
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)#这个函数产生的随机数与均值的差距不会超过两倍的标准差
    return tf.Variable(initial,dtype=tf.float32,name='weight')
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,dtype=tf.float32,name='biases')
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#strides第一个和第四个都是1，然后中间俩个代表x方向和y方向的步长,这个函数用来定义卷积神经网络
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


xs=tf.placeholder(tf.float32,[None,784])#输入是一个28*28的像素点的数据
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)

x_image=tf.reshape(xs,[-1,28,28,1])#xs的维度暂时不管，用-1表示，28,28表示xs的数据，1表示该数据是一个黑白照片，如果是彩色的，则写成3
#卷积层1
W_conv1=weight_variable([5,5,1,32])#抽取一个5*5像素，高度是32的点,每次抽出原图像的5*5的像素点，高度从1变成32
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#输出 28*28*32的图像
h_pool1=max_pool_2x2(h_conv1)##输出14*14*32的图像，因为这个函数的步长是2*2，图像缩小一半。
#卷积层2
W_conv2=weight_variable([5,5,32,64])#随机生成一个5*5像素，高度是64的点,抽出原图像的5*5的像素点，高度从32变成64
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#输出14*14*64的图像
h_pool2=max_pool_2x2(h_conv2)##输出7*7*64的图像，因为这个函数的步长是2*2，图像缩小一半。
#fully connected 1
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])#将输出的h_pool2的三维数据变成一维数据，平铺下来，（-1）代表的是有多少个例子
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#fully connected 2
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)#输出层



#开始训练数据
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))#相当于loss
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#训练函数，降低cross_entropy（loss）,AdamOptimizer适用于大的神经网络
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        batch_xs,batch_ys=mnist.train.next_batch(100)#每次从mnist数据集里面拿100个数据训练
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
        if i%50==0:
            print(compute_accuracy(mnist.test.images,mnist.test.labels))
    save_path = saver.save(sess, "my_net/save_net.ckpt")

