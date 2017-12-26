import tensorflow as tf
import numpy as np
import pickle,numpy,gzip
import datetime as dt
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc
from PIL import Image
import os
import sys
convert = ['*','X','+','-','1','2','3','4','5','6','7','8','9','0']
revert = {'*':0 ,'X':1,'+':2,'-':3,'1':4,'2':5,'3':6,'4':7,'5':8,'6':9 ,'7':10 ,'8':11,'9':12 ,'0':13}

prepath = os.path.join(os.getcwd(),'content_format_02.txt')
f2 = open(prepath)

print("Loading...")

prepath = '/media/mynewdrive/Freelancer/6100/captcha/'

dataX = []

#Taking Image no as input

I = int(input("Image No? "))
path = prepath+"%07d.png"%(I)
if os.path.exists(path):
    img = misc.imread(path).astype(np.float)
    grayim = np.dot(img[...,:3],[0.299,0.587,0.114])
    dataX.append(grayim)
else:
    print("Image doesnt exit")
    sys.exit()

labelY = f2.read().split('\n')

def Num2Char(n): 
    return convert[n];

def Char2Num(c):
    return revert[c]

def y_out(seq):
    arr = []
    for i in seq:
        arr.append(Char2Num(i))
    for i in range(0 , 7-len(seq)):
        arr.append(Char2Num('*'))
##    print(arr)
    arr2 = [0.0 for i in range(14*7)]
    for i in range(len(arr)):
        arr2[i*14+arr[i]] = 0.1428
    return arr2
    
dataY = []
for i in range(len(labelY)):
    dataY.append(y_out(labelY[i]))

# delete 
del labelY
dataY = dataY[:5957]

teX = dataX
teY = np.array(dataY[:2])


learning_rate = 0.001
batch_size = 128
batch_size_test = 1
dropout = 0.75

# define placeholder
X = tf.placeholder(tf.float32,[None,50,150,1])
Y = tf.placeholder(tf.float32,[None,98])
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev = 0.2)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.5,shape=shape)
	return tf.Variable(initial)


#probability placeholder for dropout layer
keep_prob = tf.placeholder(tf.float32,name='keep_prob')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))


##sess = tf.Session()
def conv2d(img, w, b):
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img,w,strides=[1,1,1,1],padding='SAME'),b))
def max_pool(img, k):
	return tf.nn.max_pool(img,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def make_model(_X, _weights, _biases, _dropout):
    # First we have Convolution layer 5x5x48 with relu
    conv1 = conv2d(_X,_weights['wc1'],_biases['bc1'])

    # Max Pooling (down-sampling),change input size by factor of 2
    conv1 = max_pool(conv1,k=2)

    # Apply Dropout
    conv1 = tf.nn.dropout(conv1,_dropout)

    # Then Convolution layer 5x5x64
    conv2 = conv2d(conv1,_weights['wc2'],_biases['bc2'])

    # Max Pooling 
    conv2 = max_pool(conv2,k=2)

    # Apply Dropout
    conv2 = tf.nn.dropout(conv2,_dropout)

    # Convolution layer 5x5x128
    conv3 = conv2d(conv2,_weights['wc3'],_biases['bc3'])

    # Max Pooling
    conv3 = max_pool(conv3,k=2)

    #Apply dropout
    conv3 = tf.nn.dropout(conv3,_dropout)

    #Reshape and do fully-connected hidden layer using matrix multiplication
    pool_shape= conv3.get_shape().as_list()
    fullcn = tf.reshape(conv3,[-1,pool_shape[1]* pool_shape[2]*pool_shape[3]] )

    fullcn = tf.nn.relu(
            tf.add(tf.matmul(fullcn,_weights['wd1']), _biases['bd1'])
            )
    fullcn = tf.nn.dropout(fullcn, _dropout)

    # Output
    out = tf.add(tf.matmul(fullcn,_weights['out']),_biases['out1'])

    return out

weight = {
	'wc1': tf.Variable(tf.truncated_normal([5,5,1,32],stddev = 0.1)),
	'wc2':tf.Variable(tf.truncated_normal([5,5,32,48], stddev = 0.1)),
	'wc3':tf.Variable(tf.truncated_normal([5,5,48,64], stddev = 0.1)),
	'wd1':tf.Variable(tf.truncated_normal([8512,1000], stddev = 0.1)),
	'out':tf.Variable(tf.truncated_normal([1000,98], stddev = 0.1))
}

biases = {
	'bc1': tf.Variable(0.1*tf.random_normal([32])),
	'bc2': tf.Variable(0.1*tf.random_normal([48])),
	'bc3': tf.Variable(0.1*tf.random_normal([64])),
	'bd1': tf.Variable(0.1*tf.random_normal([1000])),
	'out1': tf.Variable(0.1*tf.random_normal([98]))
}

# Construct model
predict = make_model(X,weight,biases,keep_prob)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=predict,labels=Y)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# batch, rows, cols
p = tf.reshape(predict,[-1,7,14])
# max idx acros the rows
max_idx_p = tf.argmax(p,2)

l = tf.reshape(Y,[-1,7,14])
max_idx_l = tf.argmax(l,2)

correct_pred = tf.equal(max_idx_p,max_idx_l)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

losses = list()
accuracies = list()
saver = tf.train.Saver()

def showcapt(arr):
    res = ''
    achar = -1
    for i in range(7):
        achar = arr[i]
        if achar == 0:
            return res
        else:
            res+=Num2Char(achar)
    return res 

def load(sess, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s" % ("captcha")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

def save_model(sess, checkpoint_dir, step):
    model_name = "captcha"
    model_dir = "%s" % (model_name)
    checkpoint_dir = os.path.join(checkpoint_dir,model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, os.path.join(checkpoint_dir,model_name), global_step = step)

print('\n\nTesting...')


# Launch the graph
with tf.Session() as sess:

    #load old model
    sess.run(init)
    path = os.path.join(os.getcwd(),'checkpoint')
    if load(sess, path):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")
    all_vars = tf.trainable_variables()


    step = 0
    epoch = 0
    start_epoch = dt.datetime.now()
    #training
    for step in range(1):

        # Show word predict
        pp = sess.run(predict,feed_dict={X:np.asarray(teX[:batch_size_test]).reshape(-1,50,150,1)/256.0,Y:teY[:batch_size_test],keep_prob:1.})
        p = tf.reshape(pp, [batch_size_test,7,14])
        max_idx_p = tf.argmax(p,2).eval()
        
        ff = teY[:batch_size_test]
        f = tf.reshape(ff,[batch_size_test,7,14])
        max_idx_f = tf.argmax(f,2).eval()
        print('\nprediction :')
        for tmp in range(batch_size_test):
            print(showcapt(max_idx_p[tmp]))

        ImName = "Test_images/%07d.txt"%(I)
        path = os.path.join(os.getcwd(), ImName)
        label_file = open(path, "w")
        label_file.write("Image Name = %07d.png\n"%I)
        label_file.write("Predicted Label = %s"%showcapt(max_idx_p[tmp]))
        label_file.close()
        print("\nPredicted label saved in text file");
        misc.imshow(img)
print('completed!')
sys.exit()
