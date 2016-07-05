
class args(object):
    """docstring for args"""
    def __init__(self):
        super(args, self).__init__()
    
    def __str__(self):
        fields = []
        for key, value in self.__dict__.items():
            fields.append( "%s : %s" % (str(key), str(value)) )
        return "\n".join(fields)


class KnnArgs(args):

    """docstring for  KnnArgs"""

    def __init__(self, similarity, k, model_type):
        super(KnnArgs, self).__init__()
        self.similarity = similarity
        self.k = k
        self.model_type = model_type


class SparseLinearArgs(args):

    """docstring for SparseLinearArgs"""

    def __init__(self, l1, l2, mtype, loss, penalty, intercept=False):
        super(SparseLinearArgs, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.model_type = mtype
        self.loss = loss
        self.penalty = penalty
        self.intercept = intercept


class LinearArgs(args):

    def __init__(self, l2, model_type="user"):
        super(LinearArgs, self).__init__()
        self.l2 = l2
        self.model_type = model_type


class WRMFArgs(args):

    """docstring for WRMFArgs"""

    def __init__(self, l2, nfactor, alpha=0):
        super(WRMFArgs, self).__init__()
        self.l2 = l2
        self.nfactor = nfactor
        self.alpha = alpha
