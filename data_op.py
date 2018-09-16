import numpy as np
import os

global data_full
data_full=np.array([])


def read(path=''):
    global data_full

    if os.path.isdir(path):
        for subpath in os.listdir(path):
            read(os.path.join(path,subpath))
    elif os.path.isfile(path):
        name=os.path.basename(path)
        name,_=os.path.splitext(name)
        if name=='Y':
            data=np.loadtxt(path)
            #print(data.shape)
            if data_full.shape==(0,):
                data_full=data
                #print (data_full)
            else:
                data_full=np.concatenate((data_full,data),axis=1)
                print('concatenate',' ',path)
                #print (data_full)


    else:
        print(path)

if __name__=='__main__':

    read('model/batch_data_no_validation')
    print(data_full.shape)
    np.savetxt('./model/all_data.txt',data_full)
