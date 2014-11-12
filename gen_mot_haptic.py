import numpy as np
import Image
import h5py
import os
import glob
import matplotlib.pyplot as plt
%matplotlib inline
# files:
files = glob.glob('/media/Gondor/Projects/infer-moto3D/*.png')
im = Image.open(files[0])
im_np = np.array(im).mean(axis=2).astype('float32')
files_len = len(files)
images = np.zeros([files_len,im_np.shape[0],im_np.shape[1]])
image_centers = np.zeros([files_len,3]) 
N = 30 #Scale factor
images_data = np.zeros([files_len*N,im_np.shape[0],im_np.shape[1]])
hand_location = np.zeros([files_len*N,3])
haptic = np.zeros([files_len*N,1])

for ii,fname in enumerate(files):
    im = Image.open(fname)
    images[ii] = np.array(im).mean(axis=2)
    file_nm_split = fname.split('_')
    image_centers[ii,0] = float(file_nm_split[1])
    image_centers[ii,1] = float(file_nm_split[2])
    image_centers[ii,2] = float(file_nm_split[3])
    print image_centers[ii]

def gen_haptic(centers,joints):
    dist = np.linalg.norm(centers-joints)
    if dist>1:
        haptic = 0
    else:
        haptic = 1
    return haptic

for jj,image,location in zip(xrange(files_len),images,image_centers):
    for ii in range(N):
        #Generate random hand location with in range
        idx = N*jj + ii
        if(ii<(N/2.0)):
            while(haptic[idx]==0):
                hand_location[idx] = np.random.uniform(low=-3,high=3,size=[1,3])
                haptic[idx] = gen_haptic(location,hand_location[idx])
        else:
             hand_location[idx] = np.random.uniform(low=-3,high=3,size=[1,3])
             haptic[idx] = gen_haptic(location,hand_location[idx])
        images_data[idx] = image
        
with h5py.File('firstpass_inferdata.h5','a') as fhndl:
    fhndl.create_dataset('test/images_data',data=images_data.astype('float32'),dtype='float32')
    fhndl.create_dataset('test/hand_locations',data=hand_location.astype('float32'),dtype='float32')
    fhndl.create_dataset('test/haptic',data=haptic.astype('float32'),dtype='float32')
    