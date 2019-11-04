import random
import numpy as np
class instance_selection():
    def __init__(self, model, classifier):
        self.model = model
        self.classifier = classifier
    def selection(self, x, y):
        # Standard index for selection
        standard = random.randint(0, len(x))

        # 1) Similar Probability
        prob = self.classifier.result(self.model, x)

        # probability differentiation (L2)
        d_tmp = prob[:,0] - prob[standard, 0]
        d_prob = [np.sqrt(d_tmp[i]*d_tmp[i]) for i in range(len(d_tmp))]
        prob_sorted = [d_prob[i] for i in np.argsort(d_prob)]
        print(prob_sorted)

        # 2) Similar Feature values
        # feature differentiation (L2)
        for i in range(0, 20):
            print(x.values[np.argmax(x.values, axis=0)])
        d_feature = [(x.iloc[standard] - x.iloc[i]) * (x.iloc[standard] - x.iloc[i])/
                     x.iloc[np.argmax(x.values)] for i in range(0, len(x))]
