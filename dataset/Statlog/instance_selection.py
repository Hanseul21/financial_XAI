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
        d_prob = [(prob[i,0] - prob[standard, 0]) * (prob[i, 0] - prob[standard, 0]) for i in range(0, len(x))]
        prob_sorted = np.argsort(d_prob)

        # 2) Similar Feature values
        # feature differentiation (L2)
        for i in range(0, 20):
            print(x.values[np.argmax(x.values, axis=0)])
        d_feature = [(x.iloc[standard] - x.iloc[i]) * (x.iloc[standard] - x.iloc[i])/
                     x.iloc[np.argmax(x.values)] for i in range(0, len(x))]
