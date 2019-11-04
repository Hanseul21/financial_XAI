import shap
import xgboost as xgb
from DataRead import DataPreprocessing
class SHAP:
    def shap_value(self, model, x):
        # define explainer
        self.explainer = shap.TreeExplainer(model)

        # Get shapley values
        return self.explainer.shap_values(x)

    def summary_plot(self, shap_value, x):
        shap.summary_plot(shap_value, x, feature_names= x.columns, plot_type='bar')

    def force_plot(self, shap_value, x):
        print(self.explainer.expected_value)
        shap.force_plot(self.explainer.expected_value, shap_value, feature_names=x.index, matplotlib=True)