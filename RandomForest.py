from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class RF():
    def __init__(self, data_name):
        self.data_name = data_name
        self.file_name = 'RF_'+ data_name +'.sav'
    def train(self, x, y):
        model_pth = os.path.join('Classifier', self.file_name)
        saved_model = os.path.isfile(model_pth)
        if not saved_model:
            print('Train and save RandomForest model')
            if not os.path.isdir('Classifier'):
                os.mkdir('Classifier')
            model = RandomForestClassifier(n_estimators=50, min_samples_split=4,
                                                class_weight='balanced', random_state=1, n_jobs=5)
            model.fit(x, y)
            pickle.dump(model, open(model_pth, 'wb'))
        else:
            print("saved model exists\n")
        model = pickle.load(open(model_pth, 'rb'))
        return model

    def eval(self, m, x, y):
        prob = m.score(x, y)
        print('Model accuracy : {0}'.format(prob))
        return prob

