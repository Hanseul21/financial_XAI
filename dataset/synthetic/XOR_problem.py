import numpy as np
import pandas as pd
import os
import random

##################################
#### 2-dimensional XOR problem ###
##################################
#
# Data is 5-dimensional
# only x1 and x2 is related to result y
# data is according to gaussian dist.
#
#
# Total data is 10000
# true data is 4949
# false data is 5051
#

class xor_data():
    def generate(self):
        mu = 0
        sig = 1
        tuple = 10000
        attr = 5
        tmp = 0
        eps = 1e-10
        law_data = np.ndarray(shape=(tuple, attr),dtype=float)
        law_label = np.ones((tuple),int)
        cnt = 0

        for i in range(tuple):
            for j in range(attr):
                law_data[i][j] = random.gauss(mu, sig)
                tmp += law_data[i][j]

            if law_data[i][0] - law_data[i][1] <= eps:
                law_label[i] = 1
                cnt += 1
            else:
                law_label[i] = 0

        X_train = pd.DataFrame(law_data,columns=['x1','x2','x3','x4','x5'])
        Y_train = pd.DataFrame(law_label,columns=['y'])

        train = pd.concat([X_train, Y_train],axis=1)

        print('the total number : {0} / attribute : {1} / true instance : {2}'.format(tuple, attr, cnt))
        file_name = os.path.abspath('XOR_problem')+'.csv'

        if not os.path.exists(file_name):
            train.to_csv(file_name)
            print("Generate")
        else:
            print('File already exits')