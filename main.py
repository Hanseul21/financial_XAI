from DataRead import DataPreprocessing
from XGbooster import XGB
from SHAP import SHAP
from RandomForest import RF
from dataset.Statlog.instance_selection import instance_selection
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" Classifier for financial data. --model_name='xgb' or 'rf' ")
    parser.add_argument('--model_name', type=str, default='xgb', help='Model Name')
    parser.add_argument('--data_name', type=str, default='heloc', help="Data Name")
    args = parser.parse_args()

    dp = DataPreprocessing(args.data_name)
    sh = SHAP()
    x_train, y_train = dp.read_data('train')
    x_test, y_test = dp.read_data('test')

    if args.model_name == 'xgb':
        classifier = XGB(args.data_name)
    elif args.model_name == 'rf':
        classifier = RF(args.data_name)
    else:
        print("{0} is not defined classifier. Please select 'xgb' or 'rf'".format(args.model_name))
    print('classifier is set')
    model = classifier.train(x_train, y_train)

    IS = instance_selection(model, classifier)

    IS.selection(x_test, y_test)

    # classifier.eval(model, x_test[0:], y_test[0:])
    # print(classifier.result(model, x_test[0:]))

    print('Training is ended')
    # the index of instances
    ins = 0

    # In case of rf -> shap_value shape : (2, 1046, 22) (class, ins_num, feature)
    # In case of XGB -> shap_vaule shape : (
    shap_value = sh.shap_value(model, x_test)
    print('shap value')
    print(shap_value)

    # print(np.shape(shap_value))
    sh.force_plot(shap_value[ins], x_test.iloc[ins])
    # sh.summary_plot(shap_value, x_test)
