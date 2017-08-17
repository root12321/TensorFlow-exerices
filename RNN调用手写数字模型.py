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
#print(bin)
def normalizepic(pic):
    im_arr = pic
    im_nparr = []
    for x in im_arr:
        x=1-x/255
        im_nparr.append(x)
    im_nparr = np.array([im_nparr])
    return im_nparr
#print(normalizepic(bin))
img=normalizepic(bin).reshape([1,28,28])
#print(img)
img= img.astype(np.float32)
#print(img.shape)



batch_size=1#一定要注意这个参数和训练的时候的参数不一样，训练的时候是从数据集里面每次拿128个数字进行训练，而我用自己的图片测试的时候是每次拿一张图片测试，所以batch_size=1!!!


n_inputs=28#mnist数据集是28*28像素，n_inputs代表的是每一行的28列
n_steps=28#代表28行
n_hidden_unis=128#自己定义的
n_classes=10#分类称10个，代表0-9
with tf.name_scope('inputs'):
    x=tf.placeholder(tf.float32,[None,n_steps,n_inputs],name='x_inputs')
    weights={
        'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_unis]),name='weights'),
        'out':tf.Variable(tf.random_normal([n_hidden_unis,n_classes]),name='weights1')
    }
    biases={
    #(128,)
        'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_unis,]),name='biases'),
    #(10,)
        'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]),name='biases1')
    }

def RNN(X,weights,biases):#X（128batch,28steps,,28inputs）128个数字，每个都是28行28列
    ##隐藏层
    #X(128*28,28)
    with tf.name_scope('layer'):
        X=tf.reshape(X,[-1,n_inputs])
        #X(128*28,128)
        X_in=tf.matmul(X,weights['in'])+biases['in']
        #X(128,28,128)
        X_in=tf.reshape(X_in,[-1,n_steps,n_hidden_unis])
        tf.summary.histogram('X_in', X_in)
        #cell
    with tf.name_scope('cell'):
        lstm_cell=tf.contrib.rnn.BasicLSTMCell(n_hidden_unis,forget_bias=1.0, state_is_tuple=True)
        _init_state=lstm_cell.zero_state(batch_size,dtype=tf.float32)
        outputs,states=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)
        # lstm_cell = rnn.BasicLSTMCell(n_hidden_unis)
        # outputs, states = rnn.static_rnn(lstm_cell,X_in, dtype=tf.float32)
        tf.summary.histogram('outsputs', outputs)
        tf.summary.histogram('outsputs', states)
        #输出层
    with tf.name_scope('result'):
        result=tf.matmul(states[1],weights['out'])+biases['out']
        tf.summary.histogram('result', result)
        return result
pred=RNN(x,weights,biases)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "my_net1/save_net.ckpt")
    result = tf.argmax(pred, 1)
    result = sess.run(pred, feed_dict={x: img})
    result1 = np.argmax(result, 1)
    print(result1)
