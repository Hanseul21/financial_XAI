import shap
import pandas as pd
import xgboost as xgb
from dataprocessing import DataPreprocessing as dp
class SHAP:
    def __init__(self, tr_data):
        self.tr_data = tr_data
        self.columns = tr_data.columns

    def plot_result(self, model, x):
        shap_value = self.shap_value(model, x)
        print('shap value : {0}'.format(shap_value))

        self.force_plot(shap_value, x)

        # self.summary_plot(shap_value, x)
    def shap_value(self, model, x):
        # define explainer
        # self.explainer = shap.TreeExplainer(model)
        self.explainer = shap.KernelExplainer(model.predict_proba, self.tr_data, nsamples=100, link="identity", keep_index=True)

        # Get shapley values
        return self.explainer.shap_values(x, nsamples=100)

    def summary_plot(self, shap_value, x):
        shap.summary_plot(shap_value, x, feature_names= self.columns, plot_type='bar')

    def force_plot(self, shap_value, x):
        print('expected value {0}'.format(self.explainer.expected_value))
        shap.force_plot(self.explainer.expected_value[0], shap_value[0], feature_names=self.columns, matplotlib=True)

