import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt 


class TNode(object):
    def __init__ (self, sample_idx):
        self.feat_name_or_label = None
        self.sample_idx = sample_idx
        self.sub_node = {}

class DTree(object):

    def __init__ (self, thresh):

        data = sio.loadmat("Sogou_webpage.mat")
        wordMat = data['wordMat'].astype(np.int)
        doclabel = data['doclabel'].astype(np.int) - 1 # index varies from 0~8
        wordMat = self.preprocess(wordMat) # categorize

        train_idx, test_idx = self.dataSplit(np.arange(wordMat.shape[0]))
        self.train_data = wordMat[train_idx, :]
        self.test_data = wordMat[test_idx, :]
        self.train_label = doclabel[train_idx, :] 
        self.test_label = doclabel[test_idx, :]

        self.thresh = thresh

    def preprocess(self, data):

        data = data[np.where(np.sum(data, axis=1)>0)[0], :]
        # thrs1, thrs2 = 0.001, 0.02
        # data_prcd = data / np.sum(data, axis=1)[:, np.newaxis]
        # idx1 = np.where(data_prcd <= thrs1)
        # idx2 = np.where((data_prcd>thrs1)&(data_prcd<=thrs2))
        # idx3 = np.where(data_prcd > thrs2)
        # data_prcd[idx1], data_prcd[idx2], data_prcd[idx3] = 0, 1, 2
        # return data_prcd.astype(int)
        return data

    def dataSplit(self, index):

        np.random.seed(2018)
        num_samples = index.shape[0]
        index_shuffled = np.random.choice(index, num_samples)
        # assert num_samples % 5 == 0
        one_piece = num_samples/5
        return index_shuffled[:4*one_piece], index_shuffled[4*one_piece:]

    def printTree(self, node):

        if len(node.sub_node.keys()) == 0:
            print("=== Leaf Node; Class: {} ===".format(node.feat_name_or_label))
        else:
            print("=== Branch Node ===; Feat {} ===".format(node.feat_name_or_label))
        for n in node.sub_node.values():
            self.printTree(n)
            # print("\t and its child node: {}".format(n.feat_name_or_label))

    """ generateTree
    """
    def fit(self):
        self.feat_remained = np.arange(self.train_data.shape[1])
        self.root = TNode(np.arange(self.train_data.shape[0]))
        self.splitNode(self.root, self.thresh)
        # self.printTree(self.root)

    def splitNode(self, node, thresh):

        ith, real_idx_splitted, max_delta = self.selectFeature(node)
        real_idx_splitted = dict([ (k, v) for k,v in real_idx_splitted.iteritems() if len(v)!=0 ])
        if max_delta < thresh or len(real_idx_splitted.items()) == 0:
            label_cur = np.squeeze(self.train_label[node.sample_idx])
            if len(label_cur.shape) == 0:
                label_cur = label_cur[np.newaxis]
            label_count = np.bincount(label_cur)
            label_count = label_count[np.where(label_count>0)]
            node.feat_name_or_label = np.array(list(set(label_cur.tolist())))[np.argmax(label_count)]
            return
        node.feat_name_or_label = ith
        node.sub_node = dict([ (k, TNode(v)) for k,v in real_idx_splitted.iteritems() ])
        for n in node.sub_node.values():
            self.splitNode(n, thresh)


    def selectFeature(self, node):
        
        data, label = self.train_data[node.sample_idx], self.train_label[node.sample_idx]
        label = np.squeeze(label)
        if len(label.shape) == 0:
            label = label[np.newaxis]
        root_imp = self.impurity(label)

        num = data.shape[0]
        max_delta = -np.inf
        selected = -1
        idx_splitted = []
        feat_set = range(0,2)
        for i in self.feat_remained:
            feat = data[:, i]
            idx_splitted_tmp = [ np.where(feat==f)[0] for f in feat_set ]
            label_splitted = [ label[idx] for idx in idx_splitted_tmp ]
            impurity_splitted = np.array([ self.impurity(l) for l in label_splitted ])
            impurity_splitted *= np.array([ l.shape[0] for l in label_splitted ])/float(num) 
            impurity_splitted = np.sum(impurity_splitted)
            if root_imp - impurity_splitted > max_delta:
                max_delta = root_imp - impurity_splitted
                selected = i
                idx_splitted = idx_splitted_tmp
        self.feat_remained = np.delete(self.feat_remained, np.where(self.feat_remained==selected)[0][0], axis=0)
        real_idx_splitted = dict([ (feat_set[i], node.sample_idx[idx_splitted[i]]) for i in range(len(idx_splitted)) ]) 

        return selected, real_idx_splitted, max_delta

    def impurity(self, label, select="gini"):

        label = np.squeeze(label)
        if len(label.shape) == 0:
            label = label[np.newaxis]
        if label.shape[0] == 0:
            return 0
        label_count = np.bincount(label)
        prob = label_count.astype(np.float32)/label.shape[0]

        if select == "entropy":
            imp = -np.sum(np.log(prob+1e-8)*prob)
        elif select == "gini":
            imp = 1-np.sum(prob**2)

        return imp
        
    def cross_validation(self):

        train_data = self.train_data.copy()
        train_label = self.train_label.copy()

        num_samples = self.train_data.shape[0]
        idxes = np.arange(num_samples)
        one_piece = num_samples/4
        pieces = np.array([ idxes[:one_piece], idxes[one_piece:2*one_piece], idxes[2*one_piece:3*one_piece], idxes[3*one_piece:] ]) 

        acc = np.array([])
        for i in xrange(3, -1, -1):

            print("==== running fold {}... ====".format(4-i))
            selected_pieces = pieces[list(set(range(4)) - set([i]))]
            self.train_data = train_data[np.hstack(selected_pieces), :]
            self.train_label= train_label[np.hstack(selected_pieces), :]

            val_data = train_data[pieces[i], :]
            val_label = train_label[pieces[i], :]
            self.fit()

            preds = self.decision(val_data)
            accuracy = self.accuracy(preds, np.squeeze(val_label))
            acc = np.append(acc, accuracy)
            print("accuracy: {}".format(accuracy))

        return acc

    def decision(self, items, record=False):

        preds = -np.ones(items.shape[0])
        for i, item in enumerate(items):
            node = self.root
            while not len(node.sub_node.keys()) == 0:
                if record:
                    if not hasattr(node, 'val_sample_idx'):
                        node.val_sample_idx = np.array([], dtype=int)
                    node.val_sample_idx = np.append(node.val_sample_idx, i)
                selected = node.feat_name_or_label 
                node = node.sub_node[item[selected]]

            if record:
                if not hasattr(node, 'val_sample_idx'):
                    node.val_sample_idx = np.array([], dtype=int)
                node.val_sample_idx = np.append(node.val_sample_idx, i)

            preds[i] = node.feat_name_or_label

        return preds

    def accuracy(self, preds, gts):

        assert preds.shape[0] == gts.shape[0]
        return np.sum((preds==gts))/float(preds.shape[0])

    def prune(self):

        train_data = self.train_data.copy()
        train_label = self.train_label.copy()

        num_samples = self.train_data.shape[0]
        idxes = np.arange(num_samples)
        one_piece = num_samples/4
        pieces = np.array([ idxes[:one_piece], idxes[one_piece:2*one_piece], idxes[2*one_piece:3*one_piece], idxes[3*one_piece:] ]) 

        selected_pieces = pieces[list(set(range(4)) - set([3]))]
        self.train_data = train_data[np.hstack(selected_pieces), :]
        self.train_label= train_label[np.hstack(selected_pieces), :]

        val_data = train_data[pieces[3], :]
        val_label = train_label[pieces[3], :]

        self.fit()

        preds = self.decision(val_data, record=True)
        print("Accuracy before pruned: {}".format(self.accuracy(preds, np.squeeze(val_label))))
        
        self.recursive_prune(self.root, np.squeeze(val_label))
        preds_pruned = self.decision(val_data)
        print("Accuracy after pruned: {}".format(self.accuracy(preds_pruned, np.squeeze(val_label))))

    def recursive_prune(self, node, val_label):

        if not hasattr(node, 'val_sample_idx'):
            return 0
        label_cur = np.squeeze(self.train_label[node.sample_idx])
        if len(label_cur.shape) == 0:
            label_cur = label_cur[np.newaxis]
        label_count = np.bincount(label_cur)
        label_count = label_count[np.where(label_count>0)]
        cls = np.array(list(set(label_cur.tolist())))[np.argmax(label_count)]
        pruned_acc = np.sum((val_label[node.val_sample_idx]==cls))

        if len(node.sub_node.keys()) == 0:
            return pruned_acc

        else:
            sub_acc = 0
            flag = 0
            for n in node.sub_node.values():
                sub_acc += self.recursive_prune(n, val_label)
                if len(n.sub_node.keys()) == 0:
                    flag = 1
            if sub_acc < pruned_acc and flag == 1:
                # print("pruned!")
                node.sub_node = {}
                node.feat_name_or_label = cls
            return pruned_acc



if __name__ == "__main__":


    """ select hyper params by cross-validation
    """
    thresh = np.array([0.1, 0.08, 0.05, 0.03, 0.019])
    trees = []
    acc = []
    for th in thresh:
        tree = DTree(thresh=th)
        # tree.fit()
        accuracy= tree.cross_validation()
        acc.append(accuracy.mean())
        print("Average cross-validation accuracy under threshold {}: {}".format(th, accuracy.mean()))
    
    acc = np.array(acc)
    best_th = thresh[np.argmax(acc, axis=0)] 

    print(" Best threshold is {}".format(best_th))


    print("--******* Under Threshold {} *******--".format(best_th))

    """ method1: use 4 fold of training set to train without pruned
    """
    best_tree = DTree(thresh=best_th)
    best_tree.fit()
    ## evaluate on training set
    preds_train = best_tree.decision(best_tree.train_data)
    accuracy_train = best_tree.accuracy(preds_train, np.squeeze(best_tree.train_label))
    ## evaluate on test set
    preds = best_tree.decision(best_tree.test_data)
    accuracy = best_tree.accuracy(preds, np.squeeze(best_tree.test_label))
    print("--- 4 fold training ---")
    print("Training set accuracy:{}; Test set accuracy:{}".format(accuracy_train, accuracy))

    """ method2:  use 3 fold of training set to train and prune the tree by 1 fold of validation set
    """
    print("--- 3 fold training and prune ---")
    best_tree1 = DTree(thresh=best_th)
    best_tree1.prune()
    ## evaluate on training set
    preds_train = best_tree1.decision(best_tree1.train_data)
    accuracy_train = best_tree1.accuracy(preds_train, np.squeeze(best_tree1.train_label))
    ## evaluate on test set
    preds = best_tree1.decision(best_tree1.test_data)
    accuracy = best_tree1.accuracy(preds, np.squeeze(best_tree1.test_label))
    print("Training set accuracy:{}; Test set accuracy:{}".format(accuracy_train, accuracy))

