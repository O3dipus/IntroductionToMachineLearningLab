class DecisionTreeNode:
    def __init__(self):
        self.left = None
        self.right = None
        self.depth = -1
        self.is_leaf = None
        self.dim = None
        self.threshold = None
        self.label_count = None
        self.label = None
        self.sample_size = None
        self.is_leaf = None

    # construction function of decision node
    # input :
    #       cls : self - DecisionTreeNode
    #       dim : dimension of decision property - int
    #       threshold : threshold used to make decision - float
    #       label_count : numbers of different labels in this node - float[4]
    # return :
    #       dn : node - DecisionTreeNode
    @classmethod
    def decision_node(cls, dim, threshold, label_count=None):
        dn = cls()
        dn.dim = dim
        dn.threshold = threshold
        dn.label_count = label_count
        dn.is_leaf = False
        return dn

    # construction function of label node
    # input :
    #       cls : self - DecisionTreeNode
    #       label : current node label - [1,2,3,4]
    #       sample_size : sample size - int
    # return :
    #       dn : node - DecisionTreeNode
    @classmethod
    def label_node(cls, label, sample_size=None):
        dn = cls()
        dn.label = label
        dn.sample_size = sample_size
        dn.is_leaf = True
        return dn

    def is_leaf(self):
        return self.is_leaf

    def set_depth(self, depth):
        self.depth = depth

    def set_left_child(self, left):
        self.left = left

    def set_right_child(self, right):
        self.right = right

    # labelling given data in the decision tree
    # input :
    #       data : float[8]
    # return :
    #       label : int
    def judge(self, data):
        if not self.is_leaf:
            if data[self.dim] < self.threshold:
                return self.left.judge(data)
            else:
                return self.right.judge(data)
        else:
            return self.label

    def serialize(self):
        if self.is_leaf:
            return "{} {}".format(self.is_leaf, self.label)
        else:
            return "{} {} {}".format(self.is_leaf, self.dim, self.threshold)
