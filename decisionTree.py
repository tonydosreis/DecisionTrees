import numpy as np

#tests if the x[:-1] is always associated with the same x[-1]
#in other words, if the same input always has the same output
def isDeterministic(X):
    seen = dict()
    for x in X:
        x = tuple(x)
        if x[:-1] not in seen:
            seen[x[:-1]] = x[-1]
        else:
            if( seen[x[:-1]] != x[-1] ):
                return False
    return True

def calc_entropy(P):
    #P is one dimensional array of probabilities
    result = 0
    for p in P:
        result -= p*np.log(p)

    return result

def assert_nonnegative_int(x):
    assert (type(x) == int) or (type(x) == np.int64) or (type(x) == np.int32)
    assert x >= 0

class Node():
    id = 0
    def __init__(self):
        self.id = Node.id
        Node.id += 1

class LeafNode(Node):
    def __init__(self, cla):
        super().__init__()
        assert_nonnegative_int(cla)
        self.cla = cla

    def get_class(self):
        return self.cla

    def __call__(self):
        return self.cla

    def __str__(self):
        return f"class: {self.cla}" 

class NonLeafNode(Node):

    def __init__(self, attr_index = 0):
        super().__init__()
        assert_nonnegative_int(attr_index)
        self.attr_index = attr_index
        self.children = []

    def add_child(self, child):
        assert (type(child) == NonLeafNode) or (type(child) == LeafNode)
        self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_children(self):
        return self.children

    def __call__(self, x):
        next_node_index = x[self.attr_index]

        next_node = self.children[next_node_index]

        if( type(next_node) == NonLeafNode):
            return next_node(x)
        else:
            return next_node()

    def __str__(self):
        return f"Attr: {self.attr_index}"

class DecisionTree(): 
    def __init__(self):
        self.root_node = None

    def __call__(self, x):

        #single input
        if ( len(x.shape) == 1 ):
            return self.root_node(x)
        #batch_input
        else:
            y_hat = np.zeros( (x.shape[0]), dtype = int)
            for i, x_row in enumerate(x):
                y_hat[i] = self(x_row)
            return y_hat

    def __str__(self):
        return str(self.root_node)

    def learn(self, X, Y, tokenizer):

        #Receives X and Y and selects those in X for which attr_i = poss
        #Returns Correspoding X_part and Y_part
        def partition(X,Y, attr_i, poss):
            mask  = (X[:,attr_i] == poss)
            X_part = X[mask]
            Y_part = Y[mask]
            return X_part, Y_part

        #Receives a list of elements X and calculated the probability distribution P
        def get_prob(X):
            uniques, counts = np.unique(X, return_counts=True)
            P = counts/counts.sum()
            return uniques, P

        def get_best_attr(X,Y, attr_list, tokenizer):

            tot_n = len(Y)

            best_attr = None
            min_entropy = np.inf
            final_leaf_nodes = None

            for attr_i in attr_list:
                curr_avg_entropy = 0
                curr_leaf_nodes = {}

                for attr_poss in tokenizer.getAttTokenVals(attr_i):
                    _, Y_part = partition(X,Y,attr_i,attr_poss)
                    n = len(Y_part)

                    if( len(Y_part) > 0 ):
                        uniques, P = get_prob(Y_part)
                        curr_entropy = (n/tot_n)*calc_entropy(P)
                        curr_avg_entropy += curr_entropy

                        if(curr_entropy == 0):
                            maj_class = uniques[np.argmax(P)]
                            curr_leaf_nodes[attr_poss] = maj_class

                    #Empty partition means no data points with value attr_pos for attribute attr_i
                    #node becomes leaf node with class of most probable class of parent
                    else:
                        uniques , P = get_prob(Y)
                        maj_class = uniques[np.argmax(P)]
                        curr_leaf_nodes[attr_poss] = maj_class

                if(curr_avg_entropy < min_entropy):
                    best_attr = attr_i
                    min_entropy = curr_avg_entropy
                    final_leaf_nodes = curr_leaf_nodes

            return best_attr, final_leaf_nodes

        def build_tree(node,X,Y, attr_list, tokenizer):

            attr_list = attr_list.copy()

            best_attr, leaf_nodes = get_best_attr(X,Y,attr_list, tokenizer)

            node.attr_index = best_attr
            attr_list.remove(best_attr)

            for i in tokenizer.getAttTokenVals(best_attr):
                if(i in leaf_nodes):
                    new_node = LeafNode(leaf_nodes[i])
                else:
                    new_node = NonLeafNode()
                    X_part, Y_part = partition(X,Y,best_attr,i)
                    build_tree(new_node,X_part,Y_part,attr_list, tokenizer)

                node.add_child(new_node)

        attr_list = [ i for i in range(X.shape[1])]

        self.root_node = NonLeafNode()
        build_tree(self.root_node,X,Y,attr_list, tokenizer)
        print("Done Building tree!")

    def evaluatePerformace(self, X, Y):
        Y_hat = self(X)
        temp = (Y_hat == Y)
        acc = temp.sum()/len(temp)
        return acc    

    def str_helper(node, level = 0):
        string = "\n" + "\t"*level +"-" +str(node)

        if(type(node) == NonLeafNode):
            for child in node.children:
                string += DecisionTree.str_helper(child, level + 1)

        return string

    def __str__(self):
        string = DecisionTree.str_helper(self.root_node)
        return string

def read_csv(path):
    try:
        file = open(path)
    except:
        print("Error")

    lines = file.read().split('\n')
    result =  [line.split(",") for line in lines]

    return result

class Tokenizer():

    def __init__(self, X):

        n_rows = len(X)
        n_cols = len(X[0])

        uniques = [ set() for i in range(n_cols)]

        for c in range(n_cols):
            for r in range(n_rows):
                uniques[c].add(X[r][c])

        self.word2token_dic = [ {} for i in range(len(uniques))]
        self.token2word_dic = [ {} for i in range(len(uniques))]

        for i, unique in enumerate(uniques):
            for j, word in enumerate(sorted(list(unique))):
                self.word2token_dic[i][word] = j

        for i, atrr in enumerate(self.word2token_dic):
            for key, value in atrr.items():
                self.token2word_dic[i][value] = key

    def word2token(self,X):

        n_rows = len(X)
        n_cols = len(X[0])

        tokens = np.empty( (n_rows,n_cols), dtype = int)

        for i in range(n_rows):
            for j in range(n_cols):
                tokens[i,j] = self.word2token_dic[j][X[i][j]]

        return tokens

    #given attribute index, return all possible values
    def getAttTokenVals(self, attr_i):
        return sorted(list(self.token2word_dic[attr_i].keys()))

    def __str__(self):

        string = ""

        for attr in self.word2token_dic:
            string += str(attr)
            string += "\n"

        return string

class KfoldCrossValidator():
    def __init__(self, data, k):

        self.k = k
        n_rows = data.shape[0]

        assert self.k <= n_rows

        np.random.shuffle(data)

        base_fold_sz = int(n_rows/self.k)
        add_fold_sz = n_rows % self.k
        fold_szs = [base_fold_sz + 1 if i < add_fold_sz else base_fold_sz for i in range(k) ]

        self.folds = []
        left_i = 0
        for fold_sz in fold_szs:
            right_i = left_i + fold_sz
            fold = data[left_i:right_i,:]
            self.folds.append(fold)
            left_i = right_i

    def __iter__(self):
        self.curr_i = 0
        return self
    
    def __next__(self):

        if(self.curr_i < self.k):

            test_set = self.folds[self.curr_i]
            train_set = np.empty(shape = (0, self.folds[0].shape[1]), dtype = int)

            for i in range(self.k):
                if(i != self.curr_i):
                    train_set = np.concatenate( [train_set, self.folds[i]], axis = 0 )

            self.curr_i += 1

            return train_set, test_set
            
        else:
            raise StopIteration

if __name__  == "__main__":

    data = read_csv("car_evaluation.csv")

    tokenizer = Tokenizer(data)
    data_tokens = tokenizer.word2token(data)

    n_folds = 7

    kfold = KfoldCrossValidator(data_tokens, n_folds)
    avg_test_acc = []
    print("Kold cross validation:")
    for i ,(train_set, test_set) in enumerate(kfold):
        print(f"Fold {i + 1}")

        X_train = train_set[:,:-1]
        Y_train = train_set[:,-1]

        X_test = test_set[:,:-1]
        Y_test = test_set[:,-1]

        decisionTree = DecisionTree()
        decisionTree.learn(X_train,Y_train, tokenizer)

        train_acc = decisionTree.evaluatePerformace(X_train, Y_train)
        test_acc = decisionTree.evaluatePerformace(X_test, Y_test)
        avg_test_acc.append(test_acc)

        print(f"Train set acc: {train_acc:.3f}")
        print(f"Test set acc: {test_acc:.3f}")
        print()

    print(f"Average test accuracy: {np.mean(avg_test_acc):.3f}")
    print()
    print(decisionTree)