import numpy as np

data=np.loadtxt('model/all_data.txt')
label=np.loadtxt('model/design_data/design/main.txt')

print(data.shape,label.shape)

order=range(300)

data_after=data[order]
label_after=label[order]

np.savetxt('data_dl/ob_train.txt',data_after[:250])
np.savetxt('data_dl/para_train.txt',label_after[:250])

np.savetxt('data_dl/ob_test.txt',data_after[250:])
np.savetxt('data_dl/para_test.txt',label_after[250:])

