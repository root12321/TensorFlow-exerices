import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
img = cv2.imread("D:\\opencv_camera\\digits\\5.jpg")
res=cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC)
#cv2.namedWindow("Image")
#print(img.shape)
#灰度化
emptyImage3=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

#二值化
ret, bin = cv2.threshold(emptyImage3, 100, 255,cv2.THRESH_BINARY)
cv2.imshow("a",bin)
cv2.waitKey(0)
print(bin)
def normalizepic(pic):
    im_arr = pic
    im_nparr = []
    for x in im_arr:
        x=1-x/255
        im_nparr.append(x)
    im_nparr = np.array([im_nparr])
    return im_nparr
#print(normalizepic(bin))
img=normalizepic(bin).reshape((1,784))
#print(img)
img= img.astype(np.float32)
################以上是对图片的预处理#####################


def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,dtype=tf.float32,name='weight')
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial,dtype=tf.float32,name='biases')
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

xs=tf.placeholder(tf.float32,[None,784])#输入是一个28*28的像素点的数据

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

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "my_net/save_net.ckpt")
    result= tf.argmax(prediction, 1)
    #result= result.eval(feed_dict={xs:img, keep_prob: 1.0}, session=sess)
    result=sess.run(prediction, feed_dict={xs:img,keep_prob: 1.0})
    result1 =np.argmax(result,1)
    print(result1)

    # result = sess.run(prediction, feed_dict={x: data})
    # print(result)