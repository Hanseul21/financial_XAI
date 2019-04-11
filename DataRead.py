import pandas as pd

data_pth = 'dataset/heloc/heloc_'

class DataPreprocessing():

    def __init__(self):
        self.d_type = 'train'

    def read_data(self, d_type):
        if d_type:
            self.d_type = d_type

        # Read file in csv format
        train = pd.read_csv(data_pth+'train.csv')
        test = pd.read_csv(data_pth+'test.csv')

        feature = self.get_feature(train)
        target = self.get_target((train))

        if(self.d_type == 'train'):
            return train[feature], train[target]

        elif(self.d_type == 'test'):
            return test[feature], test[target]

    def get_feature(self, x):
        return x.columns[2:]

    def get_target(self, x):
        return x.columns[1]
