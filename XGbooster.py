import xgboost as xgb

class XGB:
    def model(self, x, y):
        self.model = xgb.XGBClassifier(n_estimators=100)
        self.model.fit(x, y)
        return self.model

    def eval(self,m, x,y, d_type):
        prob = m.score(x, y)
        print('{0} result of model : {1}'.format(d_type, prob))