import h5py
import os
import numpy as np

def get_data(hdf_file,path=''):
    if isinstance(hdf_file,h5py._hl.dataset.Dataset):


        path_1 = path + hdf_file.name
        os.makedirs(path_1)
        os.rmdir(path_1)
        path_1=path_1+'.txt'
        np.savetxt(path_1,hdf_file)
        # print ('yes!')
    elif isinstance(hdf_file,h5py._hl.group.Group):
        for key in hdf_file.keys():
            path_1=path
            get_data(hdf_file[key],path_1)
        # print('no!')
    else:
        print('this type I\'ve not deal with it!')

model=h5py.File('/Users/dickie/codes/para_fit_new/exp/data.hdf','r')
groups_1=[]

get_data(model,'/Users/dickie/codes/para_fit_new/exp/data')