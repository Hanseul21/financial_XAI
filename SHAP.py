import shap
from DataRead import DataPreprocessing
class SHAP:
    def plot(self, model, x):
        # define explainer
        explainer = shap.TreeExplainer(model)

        # Get shapley values
        shap_value = explainer.shap_values(x)

        shap.summary_plot(shap_value, x, feature_names= x.columns, plot_type='bar')
        # shap.force_plot(explainer.expected_value, shap_value, x, matplotlib=True)