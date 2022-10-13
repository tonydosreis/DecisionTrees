from asyncio import _get_running_loop
import numpy as np

def assert_nonnegative_int(x):
    assert (type(x) == int) or (type(x) == np.int64) or (type(x) == np.int32)
    assert x >= 0

def isDeterministic(X):
    """tests if the x[:-1] is always associated with the same x[-1], in other words, if the same input always has the same output"""
    seen = dict()
    for x in X:
        x = tuple(x)
        if x[:-1] not in seen:
            seen[x[:-1]] = x[-1]
        else:
            if( seen[x[:-1]] != x[-1] ):
                return False
    return True

def get_prob(X):
    """Receives a list of elements X and calculates the probability distribution P"""
    uniques, counts = np.unique(X, return_counts=True)
    P = counts/counts.sum()
    return uniques, P

def calc_entropy(P):
    """Calculates entropy of discrete probability distribution P
    
    Parameters
    ----------
    P: one dimensional array of floats

    returns
    -------
    entropy: float
    """

    result = 0
    for p in P:
        result -= p*np.log(p)

    return result

def balancedAcc(y, y_hat):
    classes, counts = np.unique(y, return_counts = True)
    bacc = 0

    for cla, count in zip(classes, counts):
       temp1 = y == cla
       temp2 = y_hat == cla
       n = temp1.sum()
       acc = (temp1&temp2).sum()/n

       bacc += (1/count)*acc

    return bacc

class Node():
    id = 0
    def __init__(self):
        self.id = Node.id
        Node.id += 1

    def set_class(self, cla):
        assert_nonnegative_int(cla)
        self.cla = cla

    def get_class(self):
        return self.cla

class ClassificationNode(Node):
    def __init__(self, cla):
        super().__init__()
        self.set_class(cla)

    def add_parent(self, node):
        assert(type(node) == DecisionNode)
        self.parent = node

    def get_parent(self):
        return self.parent

    def __call__(self):
        return self.cla

class DecisionNode(Node):

    def __init__(self, split_type = "equality", attr_index = 0):
        super().__init__()
        assert_nonnegative_int(attr_index)
        assert split_type in ["equality", "inequality"]
        self.parent = None
        self.split_type = split_type
        self.attr_index = attr_index
        self.children = []

    def add_child(self, child):
        assert (type(child) == DecisionNode) or (type(child) == ClassificationNode)
        child.add_parent(self)
        self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_children(self):
        return self.children

    def add_parent(self, node):
        assert(type(node) == DecisionNode)
        self.parent = node

    def get_parent(self):
        return self.parent

    def __call__(self, x):
        if(self.split_type == "equality"):
            next_node_index = x[self.attr_index]
            next_node = self.children[next_node_index]
        else:
            if (x[self.attr_index] <= self.thresh ):
                next_node = self.children[0]
            else:
                next_node = self.children[1]

        if( type(next_node) == DecisionNode):
            return next_node(x)
        else:
            return next_node()

class DecisionTree(): 
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
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

    def learn(self, X, Y, thresh = 0, split_type = "equality"):
        """Learns decision tree from data following ID3 algorithm"""

        #Receives X and Y and selects those in X for which attr_i = poss
        #Returns Corresponding X_part and Y_part
        def partition(X,Y, attr_i, val, type = "e"):
            assert type in ["e", "le", "g"]
            if(type == "e" ):
                mask  = X[:,attr_i] == val
            elif(type == "le"):
                mask  = X[:,attr_i] <= val
            else:
                mask  = X[:,attr_i] > val
            X_part = X[mask]
            Y_part = Y[mask]
            return X_part, Y_part

        def get_best_attr_bin_split(X,Y, attr_list, thresh = 0):

            n_tot = len(Y)

            best_attr = None
            best_thresh = None
            min_entropy = np.inf
            final_leaf_nodes = None

            for attr_i in attr_list:

                for val in self.tokenizer.getAttTokenVals(attr_i)[:-1]:

                    _, Y_part_le = partition(X,Y,attr_i,val,"le")
                    _, Y_part_g = partition(X,Y,attr_i,val,"g")

                    nle = len(Y_part_le)
                    ng = len(Y_part_g)

                    if(  nle*ng > 0 ):
                        classes_le, P_le = get_prob(Y_part_le)
                        classes_g, P_g = get_prob(Y_part_g)
                        entropy_g = calc_entropy(P_g)
                        entropy_le = calc_entropy(P_le)
                        curr_entropy = (nle/n_tot)*entropy_le + (ng/n_tot)*entropy_g

                        if(curr_entropy < min_entropy):
                            min_entropy = curr_entropy
                            best_attr = attr_i
                            best_thresh = val

                            final_leaf_nodes = {}

                            if( len(attr_list) == 1 ):
                                final_leaf_nodes[0] = classes_le[np.argmax(P_le)]
                                final_leaf_nodes[1] = classes_g[np.argmax(P_g)]
                            else:  
                                if( entropy_le == 0 ):
                                    final_leaf_nodes[0] = classes_le[np.argmax(P_le)]
                                if(entropy_g == 0):
                                    final_leaf_nodes[1] = classes_g[np.argmax(P_g)]

            return best_attr, best_thresh, final_leaf_nodes


        def get_best_attr(X,Y, attr_list, thresh = 0):

            tot_n = len(Y)

            base_classes , base_P = get_prob(Y)
            base_entropy = calc_entropy(base_P)

            best_attr = None
            min_entropy = np.inf
            final_leaf_nodes = None

            for attr_i in attr_list:
                curr_avg_entropy = 0
                curr_leaf_nodes = {}

                for attr_poss in self.tokenizer.getAttTokenVals(attr_i):
                    _, Y_part = partition(X,Y,attr_i,attr_poss)
                    n = len(Y_part)

                    if( len(Y_part) > 0 ):
                        classes, P = get_prob(Y_part)

                        curr_entropy = calc_entropy(P)
                        curr_avg_entropy += (n/tot_n)*curr_entropy

                        info_gain = (n/tot_n)*(base_entropy - curr_entropy)

                        if( ( (info_gain <= thresh) and (info_gain > 0)  ) or (curr_entropy == 0) ):
                            maj_class = classes[np.argmax(P)]
                            curr_leaf_nodes[attr_poss] = maj_class

                    #Empty partition means no data points with value attr_pos for attribute attr_i
                    #node becomes leaf node with class of most probable class of parent
                    else:
                        maj_class = base_classes[np.argmax(base_P)]
                        curr_leaf_nodes[attr_poss] = maj_class

                if(curr_avg_entropy < min_entropy):
                    best_attr = attr_i
                    min_entropy = curr_avg_entropy
                    final_leaf_nodes = curr_leaf_nodes

            return best_attr, final_leaf_nodes

        def build_tree(node,X,Y, attr_list, thresh):

            #decision node class is majority class
            classes , P = get_prob(Y)

            maj_class = classes[np.argmax(P)]
            node.cla = maj_class

            attr_list = attr_list.copy()

            if(self.split_type == "equality"):
                best_attr, leaf_nodes = get_best_attr(X,Y,attr_list, thresh)
            else:
                best_attr, best_thresh, leaf_nodes = get_best_attr_bin_split(X,Y,attr_list, thresh)
                node.thresh = best_thresh

            node.attr_index = best_attr
            attr_list.remove(best_attr)

            if( self.split_type == "equality" ):
                for i in self.tokenizer.getAttTokenVals(best_attr):
                    if(i in leaf_nodes):
                        new_node = ClassificationNode(leaf_nodes[i])
                    else:
                        new_node = DecisionNode(split_type = self.split_type)
                        X_part, Y_part = partition(X,Y,best_attr,i)
                        build_tree(new_node,X_part,Y_part,attr_list, thresh)
                    node.add_child(new_node)
            else:
                for i in [0,1]:
                    if(i in leaf_nodes):
                        new_node = ClassificationNode(leaf_nodes[i])
                    else:
                        new_node = DecisionNode(split_type = self.split_type)
                        if(i == 0):
                            X_part, Y_part = partition(X,Y,best_attr,node.thresh, type = "le")
                        else:
                            X_part, Y_part = partition(X,Y,best_attr,node.thresh, type = "g")
                        build_tree(new_node,X_part,Y_part,attr_list, thresh)

                    node.add_child(new_node)

        self.split_type = split_type

        attr_list = [ i for i in range(X.shape[1])]

        self.root_node = DecisionNode(split_type = split_type)
        build_tree(self.root_node,X,Y,attr_list, thresh)
        
        print("Done Building tree!")

    def replace_node(self, old_node, new_node):
        """replaces old node in tree with new node"""

        parent_node = old_node.get_parent()
        for i, node in enumerate(parent_node.get_children()):
            if node == old_node:
                parent_node.children[i] = new_node
                new_node.add_parent(parent_node)
                break

    def prune(self, val_x, val_y):
        """currently does bottom up reduced-error pruning"""

        baseline_acc = self.evaluateAcc(val_x,val_y)

        lp = self.getLeafParents()

        while( len(lp) > 0 ):
            old_node = lp.pop()

            if(old_node.get_parent() != None):
                lp.add(old_node.get_parent())

                new_node = ClassificationNode(old_node.cla)
                self.replace_node(old_node, new_node)

                new_acc = self.evaluateAcc(val_x,val_y)

                if(new_acc < baseline_acc):
                    self.replace_node(new_node, old_node)

    def evaluateBalancedAcc(self, X, Y):
        """Calculates balanced accuracy given input and output, useful if there is class imbalance"""

        Y_hat = self(X)
        return balancedAcc(Y, Y_hat)

    def evaluateAcc(self, X, Y):
        """Calculates accuracy given input and output, useful if there is not class imbalance"""

        Y_hat = self(X)
        return (Y_hat == Y).sum()/len(Y)

    def numberOfNodes(self):
        """returns the number of nodes in the tree"""
        def numberOfNodesHelper(node):
            if( type(node) == ClassificationNode ):
                return 1
            else:
                n = 1
                for child in node.get_children():
                    n += numberOfNodesHelper(child)
                return n

        return numberOfNodesHelper(self.root_node)

    def getLeafParents(self):
        """Returns the non-leaf nodes that are parents of leaf nodes"""
        def getLeafParentsHelper(node):
            if(type(node) == ClassificationNode):
                return { node.get_parent() }
            else:
                deepestDecisionNodes = set()
                for child in node.get_children():
                    deepestDecisionNodes.update( getLeafParentsHelper(child) )

                return deepestDecisionNodes

        lp = getLeafParentsHelper(self.root_node)
        return lp

    def __str__(self):
        def strHelperEqual(node, level = 0, prefix = ""):
            if(type(node) == DecisionNode):
                string = "\n" + "\t"*level + prefix +" -> (DN) " + self.tokenizer.getAttrName(node.attr_index) + " = ?"
                for i, child in enumerate(node.children):
                    string += strHelperEqual(child, level + 1, tokenizer.convertToken2Word(i,node.attr_index) )
            else:
                string = "\n" + "\t"*level + prefix + " -> (CN) " + tokenizer.getAttrName(cla = True) + " " +  tokenizer.convertToken2Word(node.cla,  cla = True)
            return string

        def strHelperInequal(node, level = 0, prefix = ""):
            if(type(node) == DecisionNode):
                string = "\n" + "\t"*level + prefix +" -> (DN) " + self.tokenizer.getAttrName(node.attr_index) + " = ?"
                for i, child in enumerate(node.children):
                    if(i == 0):
                        string += strHelperInequal(child, level + 1, self.tokenizer.getAttrName(node.attr_index) + " <= " + tokenizer.convertToken2Word(node.thresh,node.attr_index)  )
                    else:
                        string += strHelperInequal(child, level + 1, self.tokenizer.getAttrName(node.attr_index) + " > " + tokenizer.convertToken2Word(node.thresh,node.attr_index)  )
            else:
                string = "\n" + "\t"*level + prefix + " -> (CN) " + tokenizer.getAttrName(cla = True) + " " +  tokenizer.convertToken2Word(node.cla,  cla = True)
            return string

        if(self.split_type == "equality"):
            string = strHelperEqual(self.root_node)
        else:
            string = strHelperInequal(self.root_node)
        
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
    """Converts  between categorical values and integer values(tokens)
    
    Attributes
    ----------
    word2token_dic: dict(str:int)
        dictionary with categories as keys and associated integers as values
    token2word_dic: dict(int:str)
        same as word2token_dic but with keys and values switched

    Methods
    -------
    convertWord2token( X : list of list of strings) --> list of list of numbers
        converts dataset in rows are data points and columns are features from categorical values to integer values

    getAttTokenVals(attr_index : int) --> list of ints
        given attribute index, returns all of the possible integer values of that attribute, sorted
    """

    def __init__(self, X, attr_names, word2token = None):
        """Defines an association between categories and integers.

        There are two approaches here:
        1: If word2token is specified, then the user is responsible for defining the association between categories and integers
        through the definition of word2token.
        2: If word2token is not specified then the association is learned automatically from the data. The association learned here 
        is arbitrary, the numbers are associated to the categories based on the order they appear in the row of the dataset. This 
        method is not appropriate if there is some order to the categories or if the associated numbers should capture some 
        relationship among the categorical values. In this case approach 1 is recommended. 
        
        Parameters
        ----------
            X : list of lists of strings, representing a matrix of strings
                rows represents different data points and columns represent different features.
            word2token : list of dictionaries, each dictionary has string keys and integer values - Optional, default is None
                This corresponds to the mapping between categorical values and integers. Each index of the list corresponds to
                a feature of the data X. There is a separate dictionary for each feature describing the association between
                categories (keys) and integers (values) 
        """

        self.attr_names = attr_names

        if word2token != None:
            self.word2token = word2token
            self.token2word = []
            for dic in self.word2token:
                self.token2word.append( {val : key for key, val in dic.items()} )
        else:
            n_rows, n_cols = len(X), len(X[0])
            self.word2token, self.token2word = [], []

            for c in range(n_cols):
                self.word2token.append(dict())
                self.token2word.append(dict())
                token = 0
                for r in range(n_rows):
                    curr_word = X[r][c]
                    if( curr_word not in self.word2token[c].keys()):
                        self.word2token[c][curr_word] = token
                        self.token2word[c][token] = curr_word
                        token += 1

    def getClassAttrI(self):
        return len(self.attr_names) - 1 

    def getAttrName(self, attr_i = None, cla = False):
        if(cla):
            return self.attr_names[self.getClassAttrI()]
        else:
            return self.attr_names[attr_i]

    def convertWord2token(self,X):
        """Converts data from categorical values to numerical values

        self.word2token_dic is used for converting from categories to numerical values
        
        Parameter
        ---------
            X : list of lists of strings, representing a matrix of strings
                rows represents different data points and columns represent different features
        Returns
        -------
            tokens: list of lists of int, representing matrix o integers
                rows represents different data points and columns represent different features
        """

        n_rows = len(X)
        n_cols = len(X[0])

        tokens = np.empty( (n_rows,n_cols), dtype = int)

        for i in range(n_rows):
            for j in range(n_cols):
                tokens[i,j] = self.word2token[j][X[i][j]]

        return tokens

    def convertToken2Word(self, val,attr_i = None, cla = False ):
        if(cla):
            return self.token2word[self.getClassAttrI()][val]
        else:
            return self.token2word[attr_i][val]


    def getAttTokenVals(self, attr_i):
        """given attribute index, return all possible values sorted"""

        return sorted(list(self.token2word[attr_i].keys()))

    def __str__(self):

        string = ""

        for attr in self.word2token:
            string += str(attr)
            string += "\n"

        return string

def DataDivider(X, proportions, shuffle = True):

    assert (type(proportions) == list) or (type(proportions) == tuple)
    assert sum(proportions) == 1

    if(shuffle):
        np.random.shuffle(X)

    n = X.shape[0]
    sizes = [ int(prop*n) for prop in proportions]
    rest = n - sum(sizes)

    for i in range(rest): sizes[i] += 1

    out = []

    left_i = 0
    for sz in sizes:
        right_i = sz + left_i
        out.append(X[left_i:right_i,:])
        left_i = right_i

    return out

def splitInputOutput(data):
    return data[:,:-1], data[:,-1]

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

    attr_names = { 0:"buying price", 1:"maintenance cost", 2:"number of doors", 3:"number of persons", 4:"lug boot", 5:"safety", 6:"decision" }

    word2token = [ {'low':0, 'med':1, 'high': 2, 'vhigh':3}, {'low':0, 'med':1, 'high': 2, 'vhigh':3}, {'2':0, '3':1, '4':2, '5more':3}, 
    {'2':0, '4':1, 'more':2}, {'small':0, 'med':1, 'big':2}, {'low':0, 'med':1, 'high':2}, {'unacc': 0, 'acc':1, 'good':2, 'vgood':3} ]


    tokenizer = Tokenizer(data, attr_names, word2token)
    data_tokens = tokenizer.convertWord2token(data)

    """
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

        train_acc = decisionTree.evaluatePerformance(X_train, Y_train)
        test_acc = decisionTree.evaluatePerformance(X_test, Y_test)
        avg_test_acc.append(test_acc)

        print(f"Train set acc: {train_acc:.3f}")
        print(f"Test set acc: {test_acc:.3f}")
        print()

    print(f"Average test accuracy: {np.mean(avg_test_acc):.3f}")
    print()
    """

    [train_data,val_data ,test_data] = DataDivider(data_tokens, [.5,.3,.2])

    train_x, train_y = splitInputOutput(train_data)
    val_x, val_y = splitInputOutput(val_data)
    test_x, test_y = splitInputOutput(test_data)

    decisionTree = DecisionTree(tokenizer)

    thresh = 0
    decisionTree.learn(train_x, train_y, thresh, split_type = "inequality")

    uniques, P = get_prob(test_y)

    print(f"\nBaseline acc: {100*np.max(P):.2f}%")
    print("Before prune")
    print(f"\tNumber of nodes: {decisionTree.numberOfNodes()}")
    print(f"\tTest set acc: {100*decisionTree.evaluateAcc(test_x,test_y):.2f}%")

    print(decisionTree)

    decisionTree.prune(val_x,val_y)

    print("After prune")
    print(f"\tNumber of nodes: {decisionTree.numberOfNodes()}")
    print(f"\tTest set acc: {100*decisionTree.evaluateAcc(test_x,test_y):.2f}%")

    print(decisionTree)