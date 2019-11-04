from dataprocessing import DataPreprocessing
from XGboost import XGB
from RandomForest import RF
from Explainer.SHAP import SHAP
from Explainer.global_surrogate import GS
from dataset.Statlog.instance_selection import instance_selection
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" Classifier for financial data. "
                                                 "--model='xgb' or 'rf',"
                                                 " --data='heloc' or 'statlog',"
                                                 " --explainer='shap' or 'gs' ")
    parser.add_argument('--model', type=str, default='xgb', help='Classifier Name')
    parser.add_argument('--data', type=str, default='heloc', help='Data Name')
    parser.add_argument('--explainer', type=str, default='shap', help='Explainer Name')
    args = parser.parse_args()

    dp = DataPreprocessing(args.data)

    x_train, y_train = dp.read_data('train')
    x_test, y_test = dp.read_data('test')

    if args.model == 'xgb':
        classifier = XGB(args.data)
    elif args.model == 'rf':
        classifier = RF(args.data)
    else:
        print("{0} is not defined classifier. Please select 'xgb' or 'rf'".format(args.model))
    print('classifier is set')

    if args.explainer == 'shap':
        explainer = SHAP(x_train)
    elif args.explainer == 'gs':
        explainer = GS()
    else:
        print("{0} is not defined explainer. Please select 'shap' or 'gs'".format(args.explainer))
    model = classifier.train(x_train, y_train)

    # IS = instance_selection(model, classifier)
    # IS.selection(x_test, y_test)

    # classifier.eval(model, x_test[0:], y_test[0:])
    # print(classifier.result(model, x_test[0:]))

    print('Training is ended')
    # the index of instances
    ins = 0

    # In case of rf -> shap_value shape : (2, 1046, 22) (class, ins_num, feature)
    # In case of XGB -> shap_vaule shape : (
    explainer.plot_result(model, x_test.iloc[ins:ins+1])
