import numpy as np
import scipy.sparse


class WRMF:

    def __init__(self, args):
        self.alpha = args.alpha
        self.l2 = args.l2
        self.nfactor = args.nfactor
        self.U = None
        self.V = None

    def init_params(self, m, n, U=None, V=None):
        if U is None and self.U is None:
            self.U = np.random.randn(m, self.nfactor) * 0.01
        else:
            self.U = U
        if V is None and self.V is None:
            self.V = np.random.randn(n, self.nfactor) * 0.01
        else:
            self.V = V

    def update(self, train, target_feat, X, XTX):
        n, _ = train.shape
        l2I = scipy.sparse.eye(self.nfactor) * self.l2
        for i in xrange(n):
            vec = train[i]
            nonzero_idx = vec.indices
            nonzero_value = vec.data
            tempX = X[nonzero_idx, :]
            XTCuI = (tempX.T * (nonzero_value * self.alpha))
            preinv = XTX + l2I + np.dot(XTCuI, tempX)
            inv = np.linalg.inv(preinv)
            X_R = (tempX.T * ((nonzero_value * self.alpha) + 1)).sum(axis=1)
            target_feat[i, :] = np.dot(inv, X_R)

    def updateU(self, train):
        VV = self.V.T.dot(self.V)
        self.update(train, self.U, self.V, VV)

    def updateV(self, train_T):
        UU = self.U.T.dot(self.U)
        self.update(train_T, self.V, self.U, UU)

    def fit(self, train, n_iter=10, U=None, V=None):
        assert isinstance(train, scipy.sparse.csr_matrix)
        m, n = train.shape
        train_T = train.T.tocsr()
        self.init_params(m, n, U, V)
        for _ in xrange(n_iter):
            self.updateU(train)
            self.updateV(train_T)

    def recommend_all(self, data):
        return np.dot(self.U, self.V.T)

    def recommend(self, users, data):
        ufactors = self.U[users, :]
        return np.dot(ufactors, self.V.T)

class FeatureWRMF:

    def __init__(self, args):
        self.l2 = args.l2
        self.nfactor = args.nfactor
        self.U = None
        self.V = None
        self.Wu = None
        self.Wv = None

    def initParams(self, Uf, If, U=None, V=None):

        m = Uf.shape[0]
        n = If.shape[0]
        if U is None and self.U is None:
            self.U = np.random.randn(m, self.nfactor) * 0.01
        else:
            self.U = U
        if V is None and self.V is None:
            self.V = np.random.randn(n, self.nfactor) * 0.01
        else:
            self.V = V
        self.Wu = np.random.randn(m, If.shape[1]) * 0.01
        self.Wi = np.random.randn(n, Uf.shape[1]) * 0.01

    # TODO: Fix update function
    # def update(self, X, Xf, R, Yf, Wy):
    #     Z = np.hstack([X, Xf])
    #     n = R.shape[0]
    #     I = scipy.sparse.identity(Z.shape[1], format="csr")
    #     l2I = I * self.l2
    #     ZTZ = np.dot(Z.T, Z)
    #     for i in xrange(n):
    #         vec = R[i]
    #         nonzero_idx = vec.indices
    #         nonzero_value = vec.data
    #         tempZ = Z[nonzero_idx, :]
    #         ZTCuI = (tempZ.T * (nonzero_value * self.alpha))
    #         preinv = ZTZ + l2I + np.dot(ZTCuI, tempZ)
    #         inv = np.linalg.inv(preinv)
    #         Z_R = (tempZ.T * ((nonzero_value * self.alpha) + 1)).sum(axis=1)

            
    #         target_feat[i, :] = np.dot(inv, Z_R) - 
    #         D = np.dot(np.dot(Z.T, Wy), Yf.T)        


    #     preinv = np.dot(Z.T, Z) + self.l2 * I
    #     inv = np.linalg.inv(preinv)
    #     inv_Z = np.dot(inv, Z.T)
    #     D = np.dot(np.dot(Z.T, Wy), Yf.T)        
    #     result = inv_Z * R.T - np.dot(inv, D) 
    #     return result.T
    
    # def update(self, train, target_feat, X, XTX):
    #     n, _ = train.shape
    #     l2I = scipy.sparse.eye(self.nfactor) * self.l2
    #     for i in xrange(n):
    #         vec = train[i]
    #         nonzero_idx = vec.indices
    #         nonzero_value = vec.data
    #         tempX = X[nonzero_idx, :]
    #         XTCuI = (tempX.T * (nonzero_value * self.alpha))
    #         preinv = XTX + l2I + np.dot(XTCuI, tempX)
    #         inv = np.linalg.inv(preinv)
    #         X_R = (tempX.T * ((nonzero_value * self.alpha) + 1)).sum(axis=1)
    #         target_feat[i, :] = np.dot(inv, X_R)



    def updateU(self, train, Uf, If):
        result = self.update(self.V, self.If, train, Uf, self.Wi)
        self.U, self.Wu = result[:, :self.U.shape[1]], result[:, self.U.shape[1]:]

    def updateV(self, train, Uf, If):
        result = self.update(self.U, self.Uf, train.T, If, self.Wu)
        self.V, self.Wi = result[:, :self.V.shape[1]], result[:, self.V.shape[1]:]

        
    def fit(self, train, Uf, If, n_iter=10, U=None, V=None):
        self.initParams( Uf, If, U, V)
        self.Uf = Uf
        self.If = If
        for _ in xrange(n_iter):
            self.updateU(train, Uf, If)
            self.updateV(train, Uf, If)



 
    def recommend_all(self, data):
        return np.dot(self.U, self.V.T) + np.dot(self.Uf,self.Wi.T) + np.dot(self.Wu, self.If.T )

    def recommend(self, users, data, Uf, If):
        return np.dot(self.U[users, :], self.V.T) + \
                np.dot(self.Uf[users, :], self.Wi.T) +\
                np.dot(self.Wu[users,:], self.If.T)

