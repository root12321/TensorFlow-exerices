import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
lr=0.001#学习效率
training_iters=100000#训练100000次
batch_size=128


n_inputs=28#mnist数据集是28*28像素，n_inputs代表的是每一行的28列
n_steps=28#代表28行
n_hidden_unis=128#自己定义的
n_classes=10#分类称10个，代表0-9
with tf.name_scope('inputs'):
    x=tf.placeholder(tf.float32,[None,n_steps,n_inputs],name='x_inputs')
    y=tf.placeholder(tf.float32,[None,n_classes],name='y_inputs')
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
with tf.name_scope('loss'):
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred),name='loss')
    tf.summary.scalar('loss', cost)
with tf.name_scope('train'):
    train_op=tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accurary=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
tf.summary.scalar('accurary', accurary)
init=tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs1/", sess.graph)
    sess.run(init)
    step=0
    while step*batch_size<training_iters:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs=batch_xs.reshape([batch_size,n_steps,n_inputs])
        #print(batch_xs.shape)
        sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys})
        if step%20==0:
            res=sess.run(merged,feed_dict={x:batch_xs,y:batch_ys})
            writer.add_summary(res,step)
            print(sess.run(accurary,feed_dict={x:batch_xs,y:batch_ys}))
        step+=1
    save_path = saver.save(sess, "my_net1/save_net.ckpt")