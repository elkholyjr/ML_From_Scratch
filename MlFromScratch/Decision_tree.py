import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist/len(y)
    ent = -np.sum([p * np.log2(p) for p in ps if p > 0 ])
    return ent


class Node:
    def __init__ (self, feature = None, threshold = None, left = None, right = None, value =None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self): # because the only node that will have value is the leaf node
        return self.value is not None    

class DecisionTree:

    def __init__(self, min_samples_split = 2, max_depth = 100, n_feats =None):
        self.min_samples_split =min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self,X,y):
        self.n_feats = X.shape[1] if not self.n_feats else min (self.n_feats,X.shape[1]) # to ensure it won't exceed the number of features
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y,depth=0):
        n_samples,n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split: # so this is 100% a leaf node
            leaf_value = self._most_common_label(y)
            return Node(value= leaf_value)
        
        feat_idx =  np.random.choice(n_features, self.n_feats, replace= False) # we want the array to be of length of n_feats and to choose a random from 0  to n_features and to not repeat indices

        #greedy search
        best_feat, best_thresh = self._best_criteria(X,y,feat_idx)

        left_idx, right_idx = self._split(X[:,best_feat], best_thresh)
        left = self._grow_tree(X[left_idx,:],y[left_idx], depth+1)
        right = self._grow_tree(X[right_idx,:],y[right_idx], depth+1)
        return Node(best_feat,best_thresh, left, right)
        
    def _most_common_label(self,y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0] # this return a tuple so we only need the first item and label only
        return most_common
    
    def _best_criteria(self,X,y,feat_idx):
        best_gain = -1
        split_idx,split_thresh = None,None
        for feat_idx in feat_idx:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column,threshold)
                if gain > best_gain:
                    best_gain =gain
                    split_idx =feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self,y,X_column,split_thresh):
        # parent E
        parent_entropy = entropy(y)
        #generate split
        left_idx,right_idx = self._split(X_column,split_thresh)

        if len(left_idx) == 0 or len(right_idx) ==0:
            return 0
        #weighted avg child E
        n = len(y)
        n_l,n_r = len(left_idx),len(right_idx)
        e_l,e_r = entropy(y[left_idx]),entropy(y[right_idx])
        child_entropy = (n_l/n)* e_l + (n_r/n) * e_r
        #return ig
        ig = parent_entropy -child_entropy
        return ig

    def _split(self,X_column,split_thresh):
        left_idx = np.argwhere(X_column <= split_thresh).flatten()
        right_idx = np.argwhere(X_column > split_thresh).flatten()
        return left_idx,right_idx
    
    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy:", acc)    