import scipy.io as sio
import numpy as np
import pandas as pd

mat_contents = sio.loadmat('PersonGaitDataSet.mat', squeeze_me=True)
print(mat_contents)
oct_structX = mat_contents['X']
# print(oct_struct.shape);
dfX = pd.DataFrame(data=oct_structX,index=[np.arange(48)],columns=[np.arange(321)])
#print(dfX)
oct_structY = mat_contents['Y']
dfY = pd.DataFrame(data=oct_structY,index=[np.arange(48)], columns=['0'])
#print (dfY)