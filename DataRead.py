import pandas as pd
import numpy as np
heloc_pth = 'dataset/heloc/heloc_'
stat_pth = 'dataset/Statlog/statlog_'



class DataPreprocessing():

    def __init__(self, data_name):
        self.d_type = 'train'
        self.data_name = data_name
        if(data_name == 'heloc'):
            self.data_pth = heloc_pth
        elif(data_name == 'statlog'):
            self.data_pth = stat_pth

    def read_data(self, d_type):
        if d_type:
            self.d_type = d_type

        # Read file in csv format
        train = pd.read_csv(self.data_pth+'train.csv')
        test = pd.read_csv(self.data_pth+'test.csv')

        feature = self.get_feature(train)
        target = self.get_target()

        if(self.d_type == 'train'):
            return train[feature], train[target]

        elif(self.d_type == 'test'):
            return test[feature], test[target]

    def get_feature(self, x):
        feature = [i for i in x.columns[1:] if i != self.get_target()]
        return feature

    def get_target(self):
        if self.data_name == 'heloc':
            return 'RiskPerformance'
        elif self.data_name == 'statlog':
            return 'Credit'
