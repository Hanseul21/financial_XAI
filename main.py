from DataRead import DataPreprocessing
from XGbooster import XGB
from SHAP import SHAP
import numpy as np

iter = 1000

dp = DataPreprocessing()
xgb = XGB()
sh = SHAP()

x_train, y_train = dp.read_data('train')
x_test, y_test = dp.read_data('test')

model = xgb.model(x_train, y_train)
xgb.eval(model, x_test, y_test, 'Test')

ins = x_test.iloc[0, :]

explainer = sh.plot(model, x_test)

#
# train_r = np.argsort(-shap_values_train[1])
# test_r = np.argsort(-shap_values_test[1, :])
#
# feature, target = dp.get_feature()
# print('\nshapley value of training data...')
#


# tr = 1
# for i in range(len(feature)):
#     print((i+1), ' :', feature[train_r[i]], '\t', shap_values_train[tr, train_r[i]])
#
# shap.force_plot(explainer.expected_value, shap_values_train[tr, :], x_train.iloc[tr, :], matplotlib=True)
# #plot1 = shap.force_plot(explainer.expected_value, shap_values_train, x_train)
#
#
# print('\nshapley value of test data...')
# te = 1
# for i in range(5):
#     print((i+1), ' :', feature[test_r[i]], '\t', shap_values_test[te, test_r[i]])
#
# plot2 = shap.force_plot(explainer.expected_value, shap_values_test[te, :], x_test.iloc[te, :], matplotlib=True)