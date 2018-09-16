

import tensorflow as tf
import numpy as np
from email.mime.text import MIMEText
from email.header import Header
from email.utils import parseaddr,formataddr
from email import encoders
import smtplib


def add_layer(inputs,in_size,out_size,activation_function=None,if_b=True):
	## add one more layer and return the output of this layer
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights=tf.Variable(10*tf.random_normal([in_size,out_size]),name='W')
		with tf.name_scope('biases'):
			biases=tf.Variable(tf.zeros([40,out_size])+0.1)
		with tf.name_scope('Wx_plus_b'):
			if if_b:
				Wx_plus_b=tf.matmul(inputs,Weights)+biases
			else:
				Wx_plus_b=tf.matmul(inputs,Weights)
	if activation_function is None:
		outputs=Wx_plus_b
	else:
		outputs=activation_function(Wx_plus_b)

	return outputs,Weights,biases

def add_conv(xs,dim_in,activation_function=None):##默认32层卷积核，我没写太复杂233333
	initial=tf.truncated_normal(shape=[1,5,dim_in,32],stddev=0.1)
	W=tf.Variable(initial)
	#xs=np.reshape(xs,[40,None])
	xs1=tf.expand_dims(xs,0)
	xs1=tf.expand_dims(xs1,0)
	conv=tf.nn.conv2d(xs1,W,strides=[1,1,1,1],padding='SAME')
	conv=tf.squeeze(conv)
	if activation_function is None:
		outputs=conv
	else:
		outputs=activation_function(conv)

	return outputs


def get_random_block_from_data(data0,mark0):
	num_events = data0.shape[0]
	indices = np.arange(num_events)
	np.random.shuffle(indices)

	data_x = data0[indices,:]
	data_y = mark0[indices,:]
	start_indice = np.random.randint(0,200)
	return(data_x[start_indice:start_indice+50,:],data_y[start_indice:start_indice+50,:])

def get_ordered_block_from_data(data0,mark0):
	data_x=data0.reshape([5,50,90])
	data_y=mark0.reshape([5,44,9])
	return(data_x,data_y)
x_data0 = np.loadtxt("data_dl/ob_train.txt")
#x_data0=x_data0*10
y_data0 = np.loadtxt("data_dl/para_train.txt")
#y_data0*=10
#y_data0[:,2]*=20
## define placeholder for inputs to network
with tf.name_scope('inputs'):
	xs=tf.placeholder(tf.float32,[None,90],name='x_input')
	ys=tf.placeholder(tf.float32,[None,9],name='y_input')
##　add hidden layer



xs_reshape=tf.reshape(xs,(50,90,1))
l1=tf.layers.conv1d(xs_reshape,filters=16,kernel_size=3,padding='valid',activation=tf.nn.leaky_relu)
#l1=tf.layers.dense(xs,128,activation=tf.nn.leaky_relu)
# l1=tf.layers.conv1d(l1,filters=32,kernel_size=3,padding='valid',activation=tf.nn.leaky_relu)
l1=tf.layers.average_pooling1d(l1,pool_size=3,strides=3)
l1=tf.layers.dropout(l1,rate=0.3)
print(l1)
l2=tf.layers.conv1d(l1,filters=64,kernel_size=4,activation=tf.nn.leaky_relu)
#l2=tf.layers.dense(l1,128,activation=tf.nn.leaky_relu)

l2=tf.layers.average_pooling1d(l2,pool_size=3,strides=3)
l2=tf.layers.dropout(l2,rate=0.3)

# l2=tf.layers.conv1d(l2,filters=128,kernel_size=3,activation=tf.nn.leaky_relu)
# l2=tf.layers.dropout(l2)

# prediction_reshape=tf.layers.conv1d(l2,filters=5,kernel_size=3)
# prediction=tf.reshape(prediction_reshape,(44,5))
# l2=tf.layers.dense(l2,128,activation=tf.nn.leaky_relu)
# l2=tf.layers.dropout(l2)
#l2=tf.layers.dense(l2,512,activation=tf.nn.relu,use_bias=False)
#l2=tf.layers.dropout(l2)
#l2=tf.layers.dense(l2,256,activation=tf.nn.relu,use_bias=False)
#l2=tf.layers.dropout(l2)
# l2=tf.layers.dense(l2,128,activation=tf.nn.leaky_relu)
# l2=tf.layers.dropout(l2)
#l2=tf.layers.dense(l2,64,activation=tf.nn.relu,use_bias=False)
#l2=tf.layers.dropout(l2)
## add conv layer
#l1=add_conv(l1,100,activation_function=tf.nn.relu)
## add output layer
l3=tf.reshape(l2,(50,-1))
prediction=tf.layers.dense(l3,9)
# weight=np.diag((1,1,10,1,1))
# reverse=np.diag((1,1,0.1,1,1))
# weight_tf=tf.Variable(weight,trainable=False,dtype=tf.float32)
# reverse_tf=tf.Variable(reverse,trainable=False,dtype=tf.float32)
prediction1 = tf.reshape(prediction,[-1,1])

with tf.name_scope('loss'):
	loss=tf.norm((prediction-ys),keep_dims=False)
	#loss=tf.reduce_mean(tf.reduce_sum(tf.square(tf.reshape(labels,[-1,1])-prediction1),reduction_indices=[1]))
with tf.name_scope('train'):
	train_step=tf.train.AdamOptimizer(0.00001).minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()
writer=tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)


for i in range(100000):

	x_data1, y_data1 = get_random_block_from_data(x_data0, y_data0)
		#y_data1=y_data1*10
	sess.run(train_step,feed_dict={xs:x_data1,ys:y_data1})

	if i % 1000==0:
		print(sess.run(loss,feed_dict={xs:x_data1,ys:y_data1}),i)
		#print(sess.run(ys,feed_dict={xs:x_data1,ys:y_data1}))
#y_data0[:,2]/=20
x_data1,y_data1 = get_random_block_from_data(x_data0,y_data0)

pred=sess.run(prediction,feed_dict={xs:x_data1,ys:y_data1})
print(pred)
print(y_data1)
print(sess.run(loss,feed_dict={xs:x_data1,ys:y_data1}))
x_data1,y_data1 = get_random_block_from_data(x_data0,y_data0)
y_data1=y_data1
##pred=sess.run(prediction,feed_dict={xs:x_data1,ys:y_data1})
print(sess.run(prediction,feed_dict={xs:x_data1,ys:y_data1}))
print(y_data1)
print(sess.run(loss,feed_dict={xs:x_data1,ys:y_data1}))

x_test=np.loadtxt('data_dl/ob_test.txt')
y_test=np.loadtxt('data_dl/para_test.txt')
#y_test[:,2]*=20
# x_test=np.tile(x_test,[2,1])
# y_test=np.tile(y_test,[2,1])

x_test1=x_test[0:44,:]
#x_test1*=10
y_test1=y_test[0:44,:]
#y_test1*=10
print(sess.run(prediction,feed_dict={xs:x_test,ys:y_test}))
print(y_test1)
print(sess.run(loss,feed_dict={xs:x_test,ys:y_test}))
a=sess.run(prediction,feed_dict={xs:x_test,ys:y_test})
# a=np.matmul(a,reverse)
#a[:,2]/=20
#y_test1[:,2]/=20
print(y_test[:,:])
b=(a-y_test)/y_test
c=np.zeros([50,9,2])
c[:,:,0]=y_test[:,:]
#y_test1[:,2]*=20
c[:,:,1]=b[:,:]
np.savetxt('./ob_to_p/predictions_10.txt',a[:,:])
np.savetxt('./ob_to_p/error_relative_10.txt',b[:,:])
np.set_printoptions(precision=3, suppress=True)
print(b[:,:])

msg1=MIMEText('{}\n{}'.format(sess.run(loss,feed_dict={xs:x_test,ys:y_test}),c),'plain','utf-8')
#msg2=MIMEText('{}'.format(b),'plain','utf-8')

from_addr='qstpython@163.com'
password='qst19980415'
to_addr='1600011388@pku.edu.cn'
smtp_server='smtp.163.com'
smtp_port=25
def _format_addr(s):
	name,addr=parseaddr(s)
	return formataddr((Header(name,'utf-8').encode(),addr))

msg1['From']=_format_addr('Dicky_163<%s>'%from_addr)
msg1['To']=_format_addr('Dicky_PKU<%s>'%to_addr)
msg1['Subject']=Header('Result_of_test','utf-8').encode()
server=smtplib.SMTP(smtp_server,smtp_port)
#server.starttls()
server.set_debuglevel(1)
server.login(from_addr,password)
server.sendmail(from_addr,to_addr,msg1.as_string())
server.quit()

Re=False

if Re:
	#y_data0[:,2]/=10
	x_data0/=10
	for i in range(1000000):

		x_data1, y_data1 = get_random_block_from_data(x_data0, y_data0)
		# y_data1=y_data1*10
		sess.run(train_step, feed_dict={xs: x_data1, ys: y_data1})

		if i % 10000 == 0:
			print(sess.run(loss, feed_dict={xs: x_data1, ys: y_data1}), i)
	# print(sess.run(ys,feed_dict={xs:x_data1,ys:y_data1}))
	x_data1, y_data1 = get_random_block_from_data(x_data0, y_data0)
	y_data1 = y_data1
	pred = sess.run(prediction, feed_dict={xs: x_data1, ys: y_data1})
	print(pred)
	print(y_data1)
	print(sess.run(loss, feed_dict={xs: x_data1, ys: y_data1}))
	x_data1, y_data1 = get_random_block_from_data(x_data0, y_data0)
	y_data1 = y_data1
	##pred=sess.run(prediction,feed_dict={xs:x_data1,ys:y_data1})
	print(sess.run(prediction, feed_dict={xs: x_data1, ys: y_data1}))
	print(y_data1)
	print(sess.run(loss, feed_dict={xs: x_data1, ys: y_data1}))

	x_test = np.loadtxt('./obs_normal_test.txt')
	y_test = np.loadtxt('./para_normal_test.txt')
	#y_test[:, 2] *= 10
	x_test = np.tile(x_test, [2, 1])
	y_test = np.tile(y_test, [2, 1])

	x_test1 = x_test[0:44, :]
	y_test1 = y_test[0:44, :]
	y_test1*=10

	print(sess.run(prediction, feed_dict={xs: x_test1, ys: y_test1}))
	print(y_test1)
	print(sess.run(loss, feed_dict={xs: x_test1, ys: y_test1}))
	a = sess.run(prediction, feed_dict={xs: x_test1, ys: y_test1})
	a/= 10
	y_test1 /= 10
	print(y_test1[0:23, :])
	c = np.zeros([23, 5, 2])
	c[:, :, 0] = y_test1[0:23, :]
	b = (a - y_test1) / y_test1
	c[:,:,1]=b[0:23,:]
	y_test1*= 10
	np.set_printoptions(precision=3, suppress=True)
	print(b[0:23, :])

	msg1 = MIMEText('{}\n{}'.format(sess.run(loss, feed_dict={xs: x_test1, ys: y_test1}),c), 'plain', 'utf-8')
	# msg2=MIMEText('{}'.format(b),'plain','utf-8')

	from_addr = 'qstpython@163.com'
	password = 'qst19980415'
	to_addr = '1600011388@pku.edu.cn'
	smtp_server = 'smtp.163.com'
	smtp_port = 25


	def _format_addr(s):
		name, addr = parseaddr(s)
		return formataddr((Header(name, 'utf-8').encode(), addr))


	msg1['From'] = _format_addr('Dicky_163<%s>' % from_addr)
	msg1['To'] = _format_addr('Dicky_PKU<%s>' % to_addr)
	msg1['Subject'] = Header('Result_of_test', 'utf-8').encode()
	server = smtplib.SMTP(smtp_server, smtp_port)
	# server.starttls()
	server.set_debuglevel(1)
	server.login(from_addr, password)
	server.sendmail(from_addr, to_addr, msg1.as_string())
	server.quit()