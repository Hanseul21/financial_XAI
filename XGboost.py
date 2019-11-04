import xgboost as xgb
import pickle
import os

class XGB:
    def __init__(self, data_name):
        self.data_name = data_name
        self.file_name = 'XGB_'+ self.data_name  +'.sav'
    def train(self, x, y):
        model_pth = os.path.join('Classifier', self.file_name)
        saved_model = os.path.isfile(os.path.join(model_pth, self.file_name))
        if not saved_model:
            print('Train and save XGB model')
            if not os.path.isdir('Classifier'):
                os.mkdir('Classifier')
            model = xgb.XGBClassifier(n_estimators=50)
            model.fit(x, y)
            pickle.dump(model, open(model_pth, 'wb'))
        else:
            model = pickle.load(open(model_pth, 'rb'))
        return model

    def eval(self, m, x, y):
        prob = m.score(x, y)
        print('Model accuracy : {0}'.format(prob))
        return prob

    def result(self, m, x):
        pred = m.predict_proba(x)
        return pred

