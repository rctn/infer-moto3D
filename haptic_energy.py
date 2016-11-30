#Script to evaluate how the haptic weights distribute
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from utils import tile_raster_images as tri
import theano
import numpy as np
import matplotlib.pyplot as plt
import theano.tensor as T
from pylearn2.datasets.xyzbot import xyzBot

print "Define Model"
model_path = 'sensorimotor.pkl'
model = serial.load(model_path)
train = model.dataset_yaml_src
ds = yaml_parse.load(train)
#ts = ds.get_test_set()
X, V = model.get_input_space().make_theano_batch()
layers = model.layers

print "Define y to be a bunch of haptic theano vectors"
#Make a batch of haptic vectors to solve this for
y = model.get_output_space().make_theano_batch()

print "Define costs"
rval = model.fprop((X,V))
cost = (.5*(rval-y)**2).mean()
cost_f = theano.function([X,V,y], cost)

print "Gradients"
grads = T.grad(cost, [V])
get_grad = theano.function([X,V,y], grads)

print "Initialize variables for the solver"
image = np.random.randn(1,64*64).astype('float32')
joints = np.random.randn(1, 3).astype('float32')
haptic_out = np.random.randn(1, 1).astype('float32')

print "Load Data"
ts = xyzBot('train', '/media/Gondor/Projects/infer-moto3D/firstpass_inferdata.h5')

print "For a specific image"
image = ts.X[7][np.newaxis,:]
true_joints = ts.V[7]
joints = np.array([[.1,.1,.1]]).astype('float32')
haptic_out = np.array([[1]]).astype('float32')

xx=np.arange(-3,3,.25)
yy=np.arange(-3,3,.25)
zz=np.arange(-3,3,.25)

print "Mesh grid"
xx_m,yy_m,zz_m=np.meshgrid(xx,yy,zz)
joints_batch = np.vstack((xx.flatten(),yy.flatten(),zz.flatten()))
print joints_batch.shape[0]
image = np.tile(image,[24,1])
haptic_out = np.tile(haptic_out,[24,1])
#Transpose
joints_batch = joints_batch.T
print "touching costs"
print "Shapes"
print image.shape,joints_batch.shape,haptic_out.shape

cost_batch = cost_f(image,joints_batch,haptic_out)
plt.figure()
plt.plot(cost_batch)
plt.savefig('Haptic_Cost.png')

print "outside"
xx=np.arange(3,5,.25)
yy=np.arange(3,5,.25)
zz=np.arange(3,5,.25)

print "Mesh grid"
xx_m,yy_m,zz_m=np.meshgrid(xx,yy,zz)
joints_batch = np.vstack((xx.flatten(),yy.flatten(),zz.flatten()))
print "Joints batch shape"
print joints_batch.shape
image = np.tile(image,[8,1])
haptic_out = np.tile(haptic_out,[8,1])
#Transpose
print "Outside box costs"
joints_batch = joints_batch.T
cost_batch_out = cost_f(image,joints_batch,haptic_out)
plt.figure()
plt.plot(cost_batch_out)
plt.savefig('Haptic_Cost_Out.png')
