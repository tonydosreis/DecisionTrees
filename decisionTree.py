import numpy as np

def assert_nonnegative_int(x):
    assert (type(x) == int) or (type(x) == np.int64) or (type(x) == np.int32)
    assert x >= 0

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

    def __str__(self):
        return f"class: {self.cla}" 

class DecisionNode(Node):

    def __init__(self, attr_index = 0):
        super().__init__()
        assert_nonnegative_int(attr_index)
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
        next_node_index = x[self.attr_index]

        next_node = self.children[next_node_index]

        if( type(next_node) == DecisionNode):
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

    def learn(self, X, Y, tokenizer, thresh = 0):

        #Receives X and Y and selects those in X for which attr_i = poss
        #Returns Corresponding X_part and Y_part
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

        def get_best_attr(X,Y, attr_list, tokenizer, thresh = 0):

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

                        if(curr_entropy <= thresh):
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

        def build_tree(node,X,Y, attr_list, tokenizer, thresh):

            #decision node class is majority class
            uniques , P = get_prob(Y)
            maj_class = uniques[np.argmax(P)]
            node.cla = maj_class

            attr_list = attr_list.copy()

            best_attr, leaf_nodes = get_best_attr(X,Y,attr_list, tokenizer, thresh)

            node.attr_index = best_attr
            attr_list.remove(best_attr)

            for i in tokenizer.getAttTokenVals(best_attr):
                if(i in leaf_nodes):
                    new_node = ClassificationNode(leaf_nodes[i])
                else:
                    new_node = DecisionNode()
                    X_part, Y_part = partition(X,Y,best_attr,i)
                    build_tree(new_node,X_part,Y_part,attr_list, tokenizer, thresh)

                node.add_child(new_node)

        attr_list = [ i for i in range(X.shape[1])]

        self.root_node = DecisionNode()
        build_tree(self.root_node,X,Y,attr_list, tokenizer, thresh)
        print("Done Building tree!")

    def evaluatePerformance(self, X, Y):
        Y_hat = self(X)
        temp = (Y_hat == Y)
        acc = temp.sum()/len(temp)
        return acc

    def numberOfNodes(self):
        def numberOfNodesHelper(node):
            if( type(node) == ClassificationNode ):
                return 1
            else:
                n = 1
                for child in node.get_children():
                    n += numberOfNodesHelper(child)
                return n

        return numberOfNodesHelper(self.root_node)

    def getDeepestDecisionNodes(self):
        def getDeepestDecisionNodesHelper(node, depth = 0):
            if(type(node) == ClassificationNode):
                return { (node.get_parent(),depth - 1),}
            else:
                deepestDecisionNodes = set()
                for child in node.get_children():
                    deepestDecisionNodes.update( getDeepestDecisionNodesHelper(child,depth + 1) )

                new_deepestDecisionNodes = set()
                
                max_depth = max(deepestDecisionNodes, key = lambda x : x[1])[1]

                for node, depth in deepestDecisionNodes:
                    if depth == max_depth:
                        new_deepestDecisionNodes.add( (node,depth) )

                return new_deepestDecisionNodes

        dp = getDeepestDecisionNodesHelper(self.root_node)
        return [node for node, _ in dp]

    def __str__(self):
        def strHelper(node, level = 0):
            string = "\n" + "\t"*level +"-" +str(node)

            if(type(node) == DecisionNode):
                for child in node.children:
                    string += strHelper(child, level + 1)

            return string

        string = strHelper(self.root_node)
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

    def __init__(self, X, word2token = None):
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

    word2token = [ {'low':0, 'med':1, 'high': 2, 'vhigh':3}, {'low':0, 'med':1, 'high': 2, 'vhigh':3}, {'2':0, '3':1, '4':2, '5more':3}, 
    {'2':0, '4':1, 'more':2}, {'small':0, 'med':1, 'big':2}, {'low':0, 'med':1, 'high':2}, {'unacc': 0, 'acc':1, 'good':2, 'vgood':3} ]

    tokenizer = Tokenizer(data, word2token)
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

    [train_data, test_data] = DataDivider(data_tokens, [.5,.5])
    train_x, train_y = splitInputOutput(train_data)
    test_x, test_y = splitInputOutput(test_data)

    decisionTree = DecisionTree()

    thresh = .3
    decisionTree.learn(train_x, train_y, tokenizer, thresh)
    print(decisionTree.evaluatePerformance(test_x,test_y))
    print(decisionTree.numberOfNodes())

    print(decisionTree)